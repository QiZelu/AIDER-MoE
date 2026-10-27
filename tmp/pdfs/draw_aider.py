"""Render the current AIDER architecture as editable SVG and a vector PDF."""
from pathlib import Path
from html import escape
from math import hypot
import xml.etree.ElementTree as ET

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor

ROOT = Path('/Users/zelu/Desktop/Codes/AIDER-main')
SVG = ROOT / 'figures/AIDER_architecture_20261024.svg'
PDF = ROOT / 'output/pdf/AIDER_architecture_20261024.pdf'
W, H, SCALE = 3200, 1900, .3
C = dict(ink='#24364E', arrow='#111820', blue='#3F78C6', coral='#F16B57',
         teal='#278E8A', purple='#9256BD', gold='#C48B13', grey='#708094',
         bluefill='#EFF5FF', coralfill='#FFF2EE', tealfill='#EDF9F7',
         purplefill='#F6EFFA', goldfill='#FFF8E7', white='#FFFFFF',
         bluepanel='#F8FAFF', tealpanel='#F7FBFB', goldpanel='#FFFCF5')
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
for name, file in [('T','Times New Roman.ttf'), ('TB','Times New Roman Bold.ttf'),
                   ('TI','Times New Roman Italic.ttf'), ('A','Arial.ttf'),
                   ('AB','Arial Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(FONT_DIR/file)))
pdfmetrics.registerFont(TTFont('U', '/Library/Fonts/Arial Unicode.ttf'))


class Drawing:
    def __init__(self):
        SVG.parent.mkdir(parents=True, exist_ok=True)
        PDF.parent.mkdir(parents=True, exist_ok=True)
        self.c = canvas.Canvas(str(PDF), pagesize=(W*SCALE, H*SCALE))
        self.c.setTitle('AIDER - Current Architecture')
        self.c.setSubject('Artifact prior, cross-branch FiLM, task-guided sparse expert fusion and optional MI objective')
        self.c.scale(SCALE, SCALE)
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img">',
                      '<title>AIDER architecture</title>',
                      '<desc>Current AIDER: 8 anchor frames and neighbors; frozen VITAL and SlowFast, frame-wise BLIP with trainable text and temporal self-attention; motion-compensated multiscale artifact prior; two learned artifact queries and gated residual fusion; task-private base FiLM plus shared cross-branch video residual modulation; channel concatenation to width 1536; two 12-head attention blocks, each with 7 Top-2 routed FFNs and 1 always-on shared FFN; private frame-wise heads and temporal averaging. Optional task-expert MI regularizes pre-Top-K Gate probabilities, with zero weight by default.</desc>']
        self.labels = []
        self.rect(0, 0, W, H, fill=C['white'], stroke=None, radius=0)

    def rect(self, x, y, w, h, fill='white', stroke=None, radius=16, sw=2, dash=None):
        attrs = f'x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"'
        if dash: attrs += f' stroke-dasharray="{" ".join(map(str,dash))}"'
        self.parts.append(f'<rect {attrs}/>')
        c=self.c; c.saveState(); c.setLineWidth(sw)
        if fill: c.setFillColor(HexColor(fill))
        if stroke: c.setStrokeColor(HexColor(stroke))
        if dash: c.setDash(*dash)
        c.roundRect(x,H-y-h,w,h,radius,stroke=int(bool(stroke)),fill=int(bool(fill)))
        c.restoreState()

    def polygon(self, points, fill, stroke=None, sw=2):
        self.parts.append(f'<polygon points="{" ".join(f"{x},{y}" for x,y in points)}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')
        c=self.c; c.saveState(); p=c.beginPath();p.moveTo(points[0][0],H-points[0][1])
        for x,y in points[1:]:p.lineTo(x,H-y)
        p.close();c.setLineWidth(sw)
        if fill:c.setFillColor(HexColor(fill))
        if stroke:c.setStrokeColor(HexColor(stroke))
        c.drawPath(p,stroke=int(bool(stroke)),fill=int(bool(fill)));c.restoreState()

    def line(self, points, color=None, sw=3.5, dash=None, arrow=False):
        color=color or C['arrow']
        pts=list(points)
        if arrow:
            (x0,y0),(x1,y1)=pts[-2:]
            length=hypot(x1-x0,y1-y0)
            ux,uy=(x1-x0)/length,(y1-y0)/length
            size=12 if sw>=3 else 10
            bx,by=x1-size*ux,y1-size*uy
            pts[-1]=(bx,by)
        attr=f'fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round"'
        if dash:attr += f' stroke-dasharray="{" ".join(map(str,dash))}"'
        self.parts.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in pts)}" {attr}/>')
        c=self.c;c.saveState();c.setStrokeColor(HexColor(color));c.setLineWidth(sw);c.setLineJoin(1);c.setLineCap(1)
        if dash:c.setDash(*dash)
        p=c.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,H-y)
        c.drawPath(p);c.restoreState()
        if arrow:self.polygon([(x1,y1),(bx-uy*size*.43,by+ux*size*.43),(bx+uy*size*.43,by-ux*size*.43)],color)

    def circle(self,x,y,r,fill=None,stroke=None,sw=2):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill or "none"}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')
        c=self.c;c.saveState();c.setLineWidth(sw)
        if fill:c.setFillColor(HexColor(fill))
        if stroke:c.setStrokeColor(HexColor(stroke))
        c.circle(x,H-y,r,stroke=int(bool(stroke)),fill=int(bool(fill)));c.restoreState()

    def plus(self,x,y,r=19,color=None):
        color=color or C['arrow'];self.circle(x,y,r,C['white'],color,2)
        self.line([(x-r*.55,y),(x+r*.55,y)],color,2.5)
        self.line([(x,y-r*.55),(x,y+r*.55)],color,2.5)

    def _runs(self, s, font):
        # Embed a Unicode fallback for occasional mathematical symbols.
        result=[]
        for ch in s:
            selected=font if ord(ch) in pdfmetrics.getFont(font).face.charToGlyph else 'U'
            if result and result[-1][0]==selected:result[-1]=(selected,result[-1][1]+ch)
            else:result.append((selected,ch))
        return result

    def width(self,s,size,font='T'):
        return sum(pdfmetrics.stringWidth(t,f,size) for f,t in self._runs(s,font))

    def text(self,x,y,s,size=28,font='T',color=None,anchor='middle'):
        color=color or C['ink'];runs=self._runs(s,font)
        width=sum(pdfmetrics.stringWidth(t,f,size) for f,t in runs)
        start=x-width/2 if anchor=='middle' else x-width if anchor=='end' else x
        self.labels.append((s,start,y-size,width,size*1.25))
        c=self.c;c.saveState();c.setFillColor(HexColor(color))
        for f,t in runs:
            family='Arial Unicode MS' if f=='U' else 'Arial' if f in ('A','AB') else 'Times New Roman'
            style='italic' if f=='TI' else 'normal';weight='bold' if f in ('TB','AB') else 'normal'
            self.parts.append(f'<text x="{start:.3f}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" font-style="{style}" fill="{color}">{escape(t)}</text>')
            c.setFont(f,size);c.drawString(start,H-y,t)
            start+=pdfmetrics.stringWidth(t,f,size)
        c.restoreState()

    def rich(self,x,y,s,size=28,font='TI',color=None,anchor='middle'):
        root=ET.fromstring('<r>'+s+'</r>');segments=[]
        def walk(node,fs,dy,ft):
            if node.text:segments.append((node.text,fs,dy,ft))
            for child in node:
                cfs=fs*.72 if child.tag in ('sub','sup') else fs
                cdy=dy+fs*(.25 if child.tag=='sub' else -.43 if child.tag=='sup' else 0)
                walk(child,cfs,cdy,'TB' if child.tag=='b' else ft)
                if child.tail:segments.append((child.tail,fs,dy,ft))
        walk(root,size,0,font)
        total=sum(self.width(t,fs,ft) for t,fs,dy,ft in segments)
        start=x-total/2 if anchor=='middle' else x-total if anchor=='end' else x
        for t,fs,dy,ft in segments:
            self.text(start,y+dy,t,fs,ft,color,anchor='start');start+=self.width(t,fs,ft)

    def box(self,x,y,w,h,lines,accent,fill=None,size=29,font='TB',radius=18,sw=2,dash=None):
        self.rect(x,y,w,h,fill or C['white'],accent,radius,sw,dash)
        lines=[lines] if isinstance(lines,str) else lines
        specs=[]
        for item in lines:
            s,fs,ft=(item,size,font) if isinstance(item,str) else (item+(font,))[:3] if len(item)==2 else item
            fs=min(fs,(w-24)/max(self.width(s,fs,ft),1)*fs)
            specs.append((s,fs,ft))
        step=min(size*1.12,(h-12)/len(lines));base=y+h/2-(len(lines)-1)*step/2
        for i,(s,fs,ft) in enumerate(specs):self.text(x+w/2,base+i*step+fs*.32,s,fs,ft)

    def panel(self,x,y,w,h,title,accent,fill,letter):
        self.rect(x,y,w,h,fill,accent,32,1.9,[10,7])
        self.text(x+36,y+43,f'({letter})',29,'TB',accent,anchor='start')
        self.text(x+w/2,y+44,title,38,'TB')

    def scene(self,x,y,w=120,h=82,variant=0,outline=None):
        self.rect(x,y,w,h,'#E1EDF6',outline or '#879CAC',12,1.7)
        self.rect(x+7,y+7,w-14,h*.4,'#D4E4F2',None,5)
        self.polygon([(x+w*.15,y+h*.92),(x+w*.31,y+h*.43),(x+w*.69,y+h*.43),(x+w*.85,y+h*.92)],'#A4B2C0')
        self.rect(x+w*.29,y+h*.56,w*.2,h*.18,C['ink'],None,5)
        self.rect(x+w*(.61+variant*.025),y+h*.61,w*.17,h*.13,C['coral'],None,4)

    def video(self,x,y):
        for i in (3,2,1,0):self.rect(x+i*7,y+i*6,148,100,'#D5E5F4','#738DA4',13,1.8)
        self.rect(x+7,y+58,134,34,'#93B4D3',None,5)
        self.circle(x+113,y+20,10,'#F4C463')
        self.circle(x+76,y+43,24,C['white'])
        self.polygon([(x+70,y+30),(x+70,y+56),(x+89,y+43)],C['blue'])

    def tokens(self,x,y,w=166,h=66,color=None,n=6,frame=True):
        color=color or C['blue']
        if frame:self.rect(x,y,w,h,C['white'],color,14,1.8)
        pad=11 if frame else 0;gap=5;tw=(w-2*pad-gap*(n-1))/n
        for i in range(n):
            fill=color if i%3==0 else {'#3F78C6':'#DDE9FA','#F16B57':'#F9CBC2','#278E8A':'#BFE2DF','#9256BD':'#DDCBEA'}.get(color,'#F2DEAF')
            self.rect(x+pad+i*(tw+gap),y+pad,tw,h-2*pad,fill,color,7,1.3)

    def heatmap(self,x,y,w=132,h=90):
        self.rect(x,y,w,h,C['white'],C['purple'],12,1.8)
        cols=['#F0E5F6','#DAC3E9','#B38ACE','#7F43A5','#DDC7EB','#BD9AD3','#9256B8','#EEE1F5','#E8D9F1','#D1B0E1','#A271C2','#F4EDF8']
        gap=4;cw=(w-16-3*gap)/4;ch=(h-16-2*gap)/3
        for i,fill in enumerate(cols):self.rect(x+8+(i%4)*(cw+gap),y+8+(i//4)*(ch+gap),cw,ch,fill,None,3)

    def finish(self):
        self.c.showPage();self.c.save();self.parts.append('</svg>')
        SVG.write_text('\n'.join(self.parts))
        # Check text remains within the page; detailed geometry is reviewed visually.
        outside=[(s,x,y,w,h) for s,x,y,w,h in self.labels if x<0 or y<0 or x+w>W or y+h>H]
        if outside:raise ValueError(f'Labels outside canvas: {outside}')
        print(SVG);print(PDF);print('Text runs:',len(self.labels))


d=Drawing()
# A left overview and four right-hand expansions follow the uploaded reference.
d.line([(1240,24),(1240,1870)],C['arrow'],2.5,[10,9])
d.text(610,58,'AIDER Framework Overview',44,'TB')
d.video(55,115);d.text(132,253,'Input Video',29,'TB')
d.line([(226,164),(270,164)],C['ink'],6,arrow=True)
d.box(270,112,190,105,['Uniform Sampling',('T = 8 frames',26,'TB')],C['grey'],'#F5F8FB',29)
d.line([(460,164),(520,164)],C['ink'],6,arrow=True)
d.rect(520,91,400,139,C['white'],C['blue'],22,2)
for i in range(4):d.scene(540+i*91,110,88,67,i%2,C['blue'])
d.text(720,212,'T Anchor Frames',29,'TB')
d.plus(946,160,16)
d.rect(975,100,210,130,C['purplefill'],C['purple'],22,1.8,[8,6])
for i in range(3):d.rect(992+i*59,119,56,56,['#E7D9F0','#D3B8E3','#B98ED0'][i],C['purple'],8,1.4)
d.text(1080,212,'+ T Neighbors',27,'TB',C['purple'])
d.box(45,271,250,44,['Generation Prompt p'],C['coral'],C['coralfill'],26)

# Encoder branches: the artifact prior is auxiliary, not a fourth learned encoder.
d.rect(30,350,1160,350,'#FBFDFF','#82ADEF',28,1.8,[9,7])
d.line([(720,230),(720,330),(175,330)])
d.line([(720,330),(855,330)])
for cx in (175,515,855):d.line([(cx,330),(cx,385)],arrow=True)
d.line([(1080,230),(1080,300),(1110,300),(1110,385)],arrow=True)
d.line([(45,293),(18,293),(18,441),(50,441)],C['coral'],2.6,arrow=True)
d.box(50,385,250,112,['Alignment', 'Encoder', ('BLIP + temporal SA',23,'TB'),('image frozen / text trainable',18,'T')],C['coral'],C['coralfill'],30)
d.box(390,385,250,112,['Motion','Encoder',('SlowFast',25,'TB'),('frozen video backbone',20,'T')],C['teal'],C['tealfill'],30)
d.box(730,385,250,112,['Technical','Encoder',('VITAL',25,'TB'),('frozen vision backbone',20,'T')],C['blue'],C['bluefill'],30)
d.box(1020,385,165,112,['Artifact','Prior',('(a)',24,'TB'),('non-learned',20,'T')],C['purple'],C['purplefill'],29)
for cx in (175,515,855,1102):d.line([(cx,497),(cx,558)],arrow=True)
d.tokens(92,558,166,70,C['coral'],8)
d.tokens(432,558,166,70,C['teal'],8)
d.rect(789,558,132,76,C['white'],C['blue'],12,1.8)
for row in range(3):
    for col in range(4):d.rect(800+col*29,570+row*18,23,13,['#96B7E5','#6795D3','#D4E3F6'][(row+col)%3],None,3)
d.heatmap(1040,558,128,76)
d.text(175,665,'Alignment Tokens',27,'TB',C['coral']);d.text(175,691,'8 × 768',23,'TI',C['coral'])
d.text(515,665,'Motion Tokens',27,'TB',C['teal']);d.text(515,691,'8 × 256',23,'TI',C['teal'])
d.text(855,665,'Patch Tokens',27,'TB',C['blue']);d.text(855,691,'1024D per patch',22,'TI',C['blue'])
d.rich(1102,665,'Artifact Map M<sub>τ</sub>',25,'TB',C['purple'])

# Artifact-informed technical token construction.
d.line([(855,700),(855,735)],arrow=True)
d.line([(1102,680),(1102,735)],arrow=True)
d.rect(720,735,470,230,C['bluefill'],C['blue'],30,1.8,[8,6])
d.text(955,773,'Artifact-Informed Prior',31,'TB')
d.text(955,809,'Fusion Module (b)',31,'TB')
d.box(748,832,414,49,['Prior-Guided Artifact Query'],C['purple'],C['white'],27)
d.line([(955,881),(955,899)],arrow=True)
d.box(748,899,414,49,['Gated Residual Fusion'],C['blue'],C['white'],28)

# Three FiLMs. Task conditioning is dashed gold and crosses feature lines without junctions.
d.rect(30,1000,970,196,C['tealpanel'],'#76B6B3',28,1.8,[8,6])
d.line([(175,709),(175,1040)],arrow=True)
d.line([(515,709),(515,1040)],arrow=True)
d.line([(955,965),(955,987),(855,987),(855,1040)],arrow=True)
d.box(50,1040,250,110,['Alignment','FiLM'],C['coral'],C['coralfill'],32)
d.box(390,1040,250,110,['Motion','FiLM'],C['teal'],C['tealfill'],32)
d.box(730,1040,250,110,['Technical','FiLM'],C['blue'],C['bluefill'],32)
d.text(55,1028,'FiLM (c)',26,'TB',anchor='start')
d.box(1015,1040,175,110,[('Task Description',23,'TB'),('Frozen CLIP',25,'TB'),('768D task vector',22,'TI')],C['gold'],C['goldfill'],25)
d.line([(1102,1040),(1102,1018),(260,1018)],C['gold'],2.6,[7,6])
for cx in (260,600,940):d.line([(cx,1018),(cx,1040)],C['gold'],2.6,[7,6],True)
for cx in (175,515,855):d.line([(cx,1150),(cx,1240)],arrow=True)
d.rect(50,1240,930,83,C['white'],C['grey'],18,2)
d.text(515,1272,'Channel Concatenation  ·  D = 1536',30,'TB')
for x,w,fill,txt in [(68,298,C['blue'],'Technical 512'),(366,149,C['teal'],'Motion 256'),(515,447,C['coral'],'Alignment 768')]:
    d.rect(x,1286,w,25,fill,None,4)
    d.text(x+w/2,1305,txt,21,'TB',C['white'])
d.line([(515,1323),(515,1380)],arrow=True)
d.line([(1102,1150),(1102,1380)],C['gold'],2.8,[7,6],True)
d.text(1102,1275,'task condition',22,'TI',C['gold'])
d.box(50,1380,1130,162,[('Task-Guided Attention + MoE (d)',35,'TB'),
       ('2 Transformer Blocks · 12 attention heads',29,'TB'),
       ('7 Top-2 Routed FFNs + 1 Always-On Shared FFN per block',27,'T')],C['gold'],C['goldfill'],35)
d.line([(615,1542),(615,1570),(310,1570),(310,1598)],arrow=True)
d.box(50,1598,520,109,['Final LN + Task-Private','Regression Head'],C['ink'],'#E4EEF9',32)
d.line([(570,1652),(640,1652)],arrow=True)
d.box(640,1598,250,109,['Temporal','Mean'],C['ink'],'#F2F6FB',32)
d.line([(890,1652),(960,1652)],arrow=True)
d.rect(960,1598,220,109,'#E4EEF9',C['ink'],18,2)
d.text(1070,1640,'Quality Score',31,'TB')
d.rich(1070,1683,'ŷ<sub>t</sub>',34,'TI')
for x,w,t in [(50,255,'Static'),(340,255,'Temporal'),(630,255,'Dynamic'),(920,260,'Text-Video')]:
    d.box(x,1750,w,52,[t],C['gold'],C['white'],28)
d.text(615,1732,'Four evaluation tasks',27,'TB')
d.text(615,1834,'Private: task FiLM / α / Gates / Heads.  Shared: video adapters / attention / experts.',22,'T')
d.line([(54,1870),(105,1870)],sw=3,arrow=True);d.text(120,1877,'feature flow',23,'T',anchor='start')
d.line([(410,1870),(465,1870)],C['gold'],2.5,[7,6],True);d.text(480,1877,'task condition',23,'T',anchor='start')
d.line([(825,1870),(880,1870)],C['grey'],2.5,[7,6],True);d.text(895,1877,'training only',23,'T',anchor='start')

# (a) Actual handcrafted residual, with parallel optical flow and DoG evidence.
d.panel(1270,24,1900,275,'Artifact Prior Extraction',C['purple'],C['purplefill'],'a')
d.text(1430,111,'Frame Pair',28,'TB')
d.scene(1320,130,104,78,0,C['purple']);d.scene(1440,130,104,78,1,C['purple'])
d.rich(1372,241,'I<sub>τ</sub>',27,'TI',C['purple']);d.rich(1492,241,'I<sub>τ+1</sub>',27,'TI',C['purple'])
d.box(1600,112,220,69,['Low-Pass','Optical Flow'],C['purple'],C['white'],27)
d.box(1600,211,220,65,['DoG Bands',('H1, H2, H3',24,'TI')],C['purple'],C['white'],27)
d.line([(1544,169),(1570,169),(1570,146),(1600,146)],arrow=True)
d.line([(1570,169),(1570,243),(1600,243)],arrow=True)
d.circle(1570,169,4,C['arrow'])
d.box(1885,134,300,109,['Flow-Guided Warping','+ Texture Residual'],C['purple'],C['white'],28)
d.line([(1820,146),(1850,146),(1850,165),(1885,165)],arrow=True)
d.line([(1820,243),(1850,243),(1850,212),(1885,212)],arrow=True)
d.box(2245,143,235,91,['Gaussian','Smoothing'],C['purple'],C['white'],28)
d.line([(2185,188),(2245,188)],arrow=True)
d.box(2540,143,240,91,['Relative','Normalization'],C['purple'],C['white'],28)
d.line([(2480,188),(2540,188)],arrow=True)
d.heatmap(2880,135,150,104);d.line([(2780,188),(2867,188)],arrow=True)
d.rich(2955,273,'Artifact Map M<sub>τ</sub>',30,'TB',C['purple'])
d.rich(2340,276,'Weighted residual: 0.5 R<sub>H1</sub> + 0.3 R<sub>H2</sub> + 0.2 R<sub>H3</sub>',23,'TI',C['purple'])

# (b) Projected patches feed attention; raw patches separately feed spatial pooling.
d.panel(1270,321,1900,420,'Artifact-Informed Prior Fusion', '#9DBFF0',C['bluepanel'],'b')
d.heatmap(1320,420,108,81)
d.rich(1374,529,'M<sub>τ</sub>',26,'TI',C['purple'])
d.box(1515,420,225,78,['Adaptive','Average Pool'],C['purple'],C['white'],28)
d.line([(1428,461),(1515,461)],arrow=True)
d.tokens(1790,438,135,48,C['purple'],5,False);d.line([(1740,461),(1776,461)],arrow=True)
d.rich(1858,520,'m<sub>τ</sub>',25,'TI',C['purple'])
d.box(2000,446,560,157,[('Prior-Guided Artifact Query',32,'TB'),
       ('softmax(QPᵀ / √d + λq log max(m, ε))',27,'TI')],C['purple'],C['white'],32)
d.line([(1925,461),(1970,461),(1970,483),(2000,483)],arrow=True)
d.circle(2225,405,14,'#D5BBE6',C['purple'],1.8);d.circle(2285,405,14,'#D5BBE6',C['purple'],1.8)
d.text(2255,386,'2 Learned Queries Q',25,'TB')
d.line([(2255,422),(2255,446)],arrow=True)
d.tokens(1320,558,166,54,C['blue'],6,False)
d.text(1395,632,'VITAL Patch Tokens',23,'TB',C['blue']);d.rich(1395,661,'R<sub>τ</sub> (1024D)',22,'TI',C['blue'])
d.box(1560,543,240,73,['Patch Projector',('1024 → 512',23,'TI')],C['blue'],C['white'],28)
d.line([(1486,585),(1560,585)],arrow=True)
d.tokens(1840,557,125,49,C['blue'],5,False);d.line([(1800,581),(1826,581)],arrow=True)
d.line([(1965,581),(2000,581)],arrow=True);d.rich(1902,631,'P<sub>τ</sub> (512D)',24,'TI',C['blue'])
d.line([(1510,585),(1510,683),(1560,683)],arrow=True);d.circle(1510,585,4,C['arrow'])
d.box(1560,642,295,83,['Mean Pool +','Spatial Projector / LN'],C['blue'],C['white'],27)
d.tokens(1900,662,130,43,C['blue'],5,False);d.line([(1855,683),(1887,683)],arrow=True)
d.rich(1965,733,'S<sub>τ</sub>',25,'TI',C['blue'])
d.box(2630,467,210,81,['Query','Pool'],C['purple'],C['white'],29)
d.line([(2560,507),(2630,507)],arrow=True)
d.tokens(2900,483,170,48,C['purple'],5,False);d.line([(2840,507),(2887,507)],arrow=True)
d.rich(2985,580,'A<sub>τ</sub><sup>art</sup>',28,'TI',C['purple'])
d.rect(2130,642,650,83,C['white'],C['blue'],18,2)
d.text(2455,675,'Gated Residual Fusion',31,'TB')
d.rich(2455,708,'LN(S<sub>τ</sub> + r<sub>τ</sub> ⊙ A<sub>τ</sub><sup>art</sup>)',28,'TI',C['blue'])
d.line([(2030,683),(2130,683)],arrow=True)
d.line([(2985,588),(2985,620),(2455,620),(2455,642)],arrow=True)
d.tokens(2885,661,190,45,C['blue'],6,False);d.line([(2780,683),(2872,683)],arrow=True)
d.text(2980,731,'Technical Tokens · 512D',25,'TB',C['blue'])

# (c) Implemented video_only FiLM: video residual does not read the task vector.
d.panel(1270,763,1900,415,'Cross-Branch FiLM Modulation','#72B4B0',C['tealpanel'],'c')
d.text(1500,842,'Other Two Branches Only',26,'TB')
d.rect(1320,860,170,66,C['coralfill'],C['coral'],18,2);d.rich(1405,902,'x<sub>j</sub>',31,'TI')
d.rect(1520,860,170,66,C['tealfill'],C['teal'],18,2);d.rich(1605,902,'x<sub>k</sub>',31,'TI')
d.line([(1490,893),(1505,893),(1505,942),(1715,942),(1715,892),(1740,892)],arrow=True)
d.line([(1690,893),(1740,893)],arrow=True)
d.box(1740,850,265,85,['Temporal Mean','+ Concatenation'],C['teal'],C['white'],28)
d.rect(2050,850,240,85,C['white'],C['teal'],18,2)
d.text(2170,882,'Branch Summary MLP',22,'TB');d.rich(2170,918,'u<sub>b</sub> · 256D',25,'TI')
d.line([(2005,893),(2050,893)],arrow=True)
d.box(2340,850,290,85,['Video Residual MLP',('shared across tasks',23,'T')],C['blue'],C['white'],28)
d.line([(2290,893),(2340,893)],arrow=True)
d.rect(2700,850,400,85,C['bluefill'],C['blue'],18,2)
d.rich(2900,904,'Δγ<sub>b</sub>(v),  Δβ<sub>b</sub>(v)',31,'TI')
d.line([(2630,893),(2700,893)],arrow=True)
d.rect(1320,987,240,73,C['goldfill'],C['gold'],18,2)
d.text(1440,1013,'Fixed Task Embedding',22,'TB');d.rich(1440,1045,'e<sub>t</sub> · 768D',27,'TI')
d.box(1630,986,300,73,['Task-Private Affine'],C['gold'],C['white'],28)
d.line([(1560,1023),(1630,1023)],C['gold'],2.8,[7,6],True)
d.rect(1990,986,260,73,C['goldfill'],C['gold'],18,2)
d.rich(2120,1035,'γ<sub>b</sub>(t),  β<sub>b</sub>(t)',30,'TI')
d.line([(1930,1023),(1990,1023)],C['gold'],2.8,arrow=True)
d.box(1630,1090,300,60,['Task-Private α Gate'],C['gold'],C['white'],28)
d.line([(1440,1060),(1440,1120),(1630,1120)],C['gold'],2.8,[7,6],True)
d.rect(1990,1090,260,60,C['goldfill'],C['gold'],18,2)
d.rich(2120,1130,'α<sub>t</sub> = sigmoid(g<sub>t</sub>)',27,'TI')
d.line([(1930,1120),(1990,1120)],C['gold'],2.8,arrow=True)
d.text(2120,1170,'one scalar shared across branches',20,'T',C['gold'])
d.rect(2395,960,270,42,C['white'],C['blue'],12,2)
d.rich(2530,990,'Target Branch x<sub>b</sub>',25,'TI')
d.rect(2370,1020,745,130,C['white'],C['blue'],18,2)
d.text(2742,1054,'FiLM Modulation',31,'TB')
d.rich(2742,1094,'x′<sub>b</sub> = (1 + γ<sub>b</sub>(t) + α<sub>t</sub> Δγ<sub>b</sub>(v)) ⊙ x<sub>b</sub>',27,'TI')
d.rich(2742,1134,'+ β<sub>b</sub>(t) + α<sub>t</sub> Δβ<sub>b</sub>(v)',27,'TI')
d.line([(2530,1002),(2530,1020)],arrow=True)
d.line([(2900,935),(2900,1020)],arrow=True)
d.line([(2250,1023),(2305,1023),(2305,1070),(2370,1070)],C['gold'],2.8,arrow=True)
d.line([(2250,1120),(2370,1120)],C['gold'],2.8,arrow=True)
d.text(2742,1170,'Same modulation coefficients broadcast over all T frames',23,'T',C['grey'])

# (d) Exactly two blocks: attention residual, sparse FFNs and always-on shared FFN.
d.panel(1270,1200,1900,670,'Task-Guided Attention + MoE Fusion',C['gold'],C['goldpanel'],'d')
d.text(1426,1285,'Concatenated Tokens',28,'TB')
d.tokens(1320,1305,215,54,C['blue'],8,False)
d.text(1426,1390,'8 × 1536',26,'TI')
d.plus(1585,1332,20)
d.line([(1535,1332),(1565,1332)],arrow=True)
d.box(1318,1412,224,64,['Temporal Position','Encoding'],C['gold'],C['white'],27)
d.line([(1542,1444),(1585,1444),(1585,1352)],arrow=True)
d.rect(1640,1285,1360,370,C['white'],C['gold'],25,1.7,[9,6])
d.text(2320,1321,'Transformer Block × 2',32,'TB')
d.line([(1605,1332),(1660,1332),(1660,1455),(1740,1455)],arrow=True)
d.box(1740,1410,280,90,['LayerNorm +','12-Head Self-Attention'],C['gold'],C['goldfill'],28)
d.plus(2095,1455,20)
d.line([(2020,1455),(2075,1455)],arrow=True)
d.line([(1685,1455),(1685,1355),(2095,1355),(2095,1435)],arrow=True)
d.circle(1685,1455,4,C['arrow'])
d.line([(2115,1455),(2760,1455)],arrow=True)
d.circle(2170,1455,4,C['arrow']);d.rich(2150,1440,'u',26,'TI')
d.box(2260,1350,340,90,['Always-On','Shared FFN'],C['gold'],C['goldfill'],29)
d.line([(2170,1455),(2170,1395),(2260,1395)],arrow=True)
d.line([(2600,1395),(2780,1395),(2780,1435)],arrow=True)
d.rect(2260,1510,340,100,C['goldfill'],C['gold'],17,2)
d.text(2430,1540,'7 Routed FFNs · Top-2',27,'TB')
for i in range(7):
    x=2278+i*45;active=i in (1,4)
    d.rect(x,1561,33,31,C['gold'] if active else '#FAEDD0',C['gold'],5,1.2)
    d.text(x+16.5,1583,f'E{i+1}',18,'TB',C['white'] if active else C['gold'])
d.line([(2170,1455),(2170,1560),(2260,1560)],arrow=True)
d.line([(2600,1560),(2780,1560),(2780,1475)],arrow=True)
d.plus(2780,1455,20)
d.rich(2840,1437,'y',27,'TI')
d.rect(1318,1532,224,78,C['goldfill'],C['gold'],18,2)
d.text(1430,1563,'Fixed Task',27,'TB');d.rich(1430,1597,'Embedding e<sub>t</sub>',27,'TB')
d.box(1740,1535,285,70,['Task-Private Gate',('768 → 7 logits',23,'TI')],C['gold'],C['white'],28)
d.line([(1542,1571),(1740,1571)],C['gold'],2.8,[7,6],True)
d.line([(2025,1570),(2110,1570),(2110,1531),(2260,1531)],C['gold'],2.8,[7,6],True)
d.line([(1882,1605),(1882,1630),(1360,1630),(1360,1805)],C['grey'],2.4,[7,6],True)
d.rich(1567,1622,'p(e | t), before Top-K',21,'TI',C['grey'])
d.line([(2800,1455),(3070,1455),(3070,1678),(1510,1678),(1510,1747),(1580,1747)],arrow=True)
d.box(1580,1710,160,75,['Final','LayerNorm'],C['ink'],C['white'],27)
d.line([(1740,1747),(1790,1747)],arrow=True)
d.box(1790,1695,330,90,[('Task-Private Head',28,'TB'),('Frame-Wise MLP',25,'TB'),('1536 → 256 → 64 → 16 → 1',21,'TI')],C['ink'],'#E9F1FB',28)
d.tokens(2190,1724,150,43,C['blue'],8,False);d.line([(2120,1747),(2177,1747)],arrow=True)
d.text(2265,1786,'8 Frame Scores',25,'TB')
d.box(2400,1710,230,75,['Temporal Mean'],C['ink'],C['white'],29)
d.line([(2340,1747),(2400,1747)],arrow=True)
d.rect(2700,1710,180,75,'#E4EEF9',C['ink'],18,2)
d.rich(2790,1740,'Score ŷ<sub>t</sub>',29,'TB');d.text(2790,1771,'per video',23,'T')
d.line([(2630,1747),(2700,1747)],arrow=True)
d.text(3024,1740,'One score per',25,'T');d.text(3024,1772,'video and task',25,'T')
d.rect(1300,1805,1840,45,'#F5F7FA',C['grey'],13,1.5,[7,6])
d.rich(2220,1836,'Optional training: L = L<sub>SmoothL1</sub> − λ<sub>MI</sub>(I<sub>1</sub> + I<sub>2</sub>)/2  |  P(t)=1/4; pre-Top-K Softmax; λ<sub>MI</sub>=0 by default',25,'T',C['ink'])
d.line([(2790,1785),(2790,1805)],C['grey'],2.4,[7,6],True)
d.finish()
