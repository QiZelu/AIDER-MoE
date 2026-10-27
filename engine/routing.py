"""当前两层共享专家的路由统计与固定任务表导出。"""

import copy
from engine.io import write_json, write_csv
import itertools
import torch
from models.registry import TASKS


@torch.no_grad()
def initial_task_routes(model):
    task_order = tuple(model.task['order']) if hasattr(model, 'task') else TASKS
    condition = model.dimension_encoder(
        torch.arange(len(task_order), device=next(model.parameters()).device)
    )
    report = {}
    for name, pool in model.pools():
        shared = name.endswith(".shared")
        tasks = task_order if shared else (name.split(".")[-1],)
        routes = {}
        for task in tasks:
            index = task_order.index(task)
            kwargs = ({'task_ids': torch.tensor([index], device=condition.device)}
                      if getattr(model, 'private_task_modules', False) else {})
            r = pool.route(condition[index:index+1], **kwargs)
            routes[task] = {k: v[0].cpu().tolist() for k, v in r.items()}
        intersections, jaccard = {}, {}
        if shared:
            for a, b in itertools.product(task_order, repeat=2):
                x, y = set(routes[a]["topk_indices"]), set(routes[b]["topk_indices"])
                intersections[f"{a}:{b}"] = len(x & y)
                jaccard[f"{a}:{b}"] = len(x & y) / len(x | y)
        report[name] = dict(
            shared=shared,
            tasks=routes,
            intersection=intersections,
            jaccard=jaccard,
            num_experts=pool.num_experts,
        )
        if getattr(model, 'private_task_modules', False):
            report[name].update(gate_policy='task_private', expert_policy='cross_task_shared',
                              gate_identity_by_task={t: name + '.gate.' + t for t in tasks})
    return dict(
        variant=model.variant,
        film_video_condition=model.film_video_condition,
        pools=report,
        note="Expert IDs in different pools refer to different parameters; intersections only compare one shared pool.",
    )


@torch.no_grad()
def task_routes(model):
    tasks = tuple(model.task['order'])
    report = initial_task_routes(model)
    for name, pool in report["pools"].items():
        initial = model.initial_task_routes["pools"][name]["tasks"]
        sets = {t: set(pool["tasks"][t]["topk_indices"]) for t in tasks}
        pool["active_union"] = sorted(set.union(*sets.values()))
        pool["active_expert_count"] = len(pool["active_union"])
        pool["unused_experts"] = sorted(
            set(range(pool["num_experts"])) - set(pool["active_union"])
        )
        pool["mean_pair_intersection"] = (
            sum(len(sets[a] & sets[b]) for a, b in itertools.combinations(tasks, 2))
            / (len(tasks) * (len(tasks) - 1) / 2)
        )
        pool["changes_from_initialization"] = {
            t: dict(
                added=sorted(sets[t] - set(initial[t]["topk_indices"])),
                removed=sorted(set(initial[t]["topk_indices"]) - sets[t]),
                retained=sorted(sets[t] & set(initial[t]["topk_indices"])),
            )
            for t in tasks
        }
    report["task_embedding_transform"] = model.experiment["task_embedding_transform"]
    return report


def export_encoding(model, out):
    tasks = tuple(model.task['order'])
    table = model.dimension_encoder
    matrices = {}
    for label, value in (
        ("original", table.original_feature),
        ("centered_normalized", table.encoded_feature),
    ):
        value = value.float().cpu()
        cosine = (
            torch.nn.functional.normalize(value, dim=1)
            @ torch.nn.functional.normalize(value, dim=1).T
        )
        matrices[label] = cosine.tolist()
        write_csv(
            out / "encoding" / f"{label}_cosine.csv",
            [
                dict(task=t, **dict(zip(tasks, cosine[i].tolist())))
                for i, t in enumerate(tasks)
            ],
        )
    write_json(
        out / "encoding" / "task_tables.json",
        dict(
            task_order=list(tasks),
            descriptions=copy.deepcopy(model.task["descriptions"]),
            original=table.original_feature.cpu().tolist(),
            mean=table.task_mean.cpu().tolist(),
            centered_norms=table.centered_norms.cpu().tolist(),
            transformed=table.encoded_feature.cpu().tolist(),
            hashes=model.task_embedding_hashes,
            cosine_matrices=matrices,
        ),
    )
    write_json(
        out / "encoding" / "original_input_initial_routes.json",
        model.original_input_initial_routes,
    )


class RouteCounts:
    def __init__(self, model):
        self.model = model
        self.tasks = tuple(model.task['order']) if hasattr(model, 'task') else TASKS
        self.counts = {
            n: torch.zeros(len(self.tasks), p.num_experts, dtype=torch.long) for n, p in model.pools()
        }

    def add(self, ids, routing):
        ids = ids.detach().cpu()
        if set(routing["pools"]) != set(self.counts):
            raise ValueError("Routing pool coverage mismatch")
        for n, c in self.counts.items():
            selected = routing["pools"][n]["topk_indices"].detach().cpu()
            for t in range(len(self.tasks)):
                c[t] += torch.bincount(
                    selected[ids == t].flatten(), minlength=c.shape[1]
                )

    def report(self):
        return dict(
            variant=self.model.variant,
            pools={
                n: dict(
                    activation_counts_by_task=dict(zip(self.tasks, c.tolist())),
                    total_activation_counts=c.sum(0).tolist(),
                    unused_experts=torch.where(c.sum(0) == 0)[0].tolist(),
                )
                for n, c in self.counts.items()
            },
        )
