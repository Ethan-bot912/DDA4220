from pathlib import Path
import json,re,html
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage

P=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('Song','C:/Windows/Fonts/simsun.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('Hei','C:/Windows/Fonts/simhei.ttf'))
normal=ParagraphStyle('Body',fontName='Song',fontSize=10,leading=16,wordWrap='CJK',spaceAfter=9,textColor=colors.black)
small=ParagraphStyle('Cell',parent=normal,fontSize=8.2,leading=11,spaceAfter=0)
head=ParagraphStyle('Heading',parent=normal,fontName='Hei',fontSize=17,leading=23,spaceAfter=15)
title=ParagraphStyle('Title',parent=head,fontSize=23,leading=32,spaceAfter=14)
caption=ParagraphStyle('Caption',parent=normal,fontSize=8.5,leading=12,alignment=TA_CENTER,spaceAfter=10)
def para(s,style=normal):return Paragraph(html.escape(s),style)
width=A4[0]-96
story=[]
def foot(c,d):
 c.setFont('Song',8);c.setFillColor(colors.HexColor('#555555'));c.drawCentredString(A4[0]/2,25,str(d.page))
for i,(heading,body) in enumerate(json.loads((P/'report_sections.json').read_text(encoding='utf-8'))):
 if i:story.append(PageBreak())
 else:
  story += [para('灾前灾后卫星影像建筑损毁分类实验报告',title),para('xBD 四事件子集上的 ResNet18 单种子受控对比'),para('报告日期 2026年10月15日    验证结果版    seed42'),Spacer(1,8)]
 story.append(para(heading,head))
 for block in body.split('\n\n'):
  if block.startswith('|'):
   rows=[[s.strip() for s in line.strip('|').split('|')] for line in block.splitlines() if not re.match(r'^\|[\s|:-]+\|$',line)]
   n=len(rows[0]);weights=[1]*n
   if n==2:weights=[1,2.8]
   if n==3:weights=[1,3,1.2]
   if rows[0][0]=='阶段':weights=[1,2.8,1.5]
   if n==6:weights=[1.2,.7,1.1,1.1,1.1,1.1]
   data=[[para(v,small) for v in row] for row in rows]
   t=Table(data,colWidths=[width*w/sum(weights) for w in weights],repeatRows=1,hAlign='LEFT')
   t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DDE7EF')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F9')]),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
   story += [t,Spacer(1,11)]
  elif block.startswith('!['):
   label,path=re.match(r'!\[(.*?)\]\((.*?)\)',block).groups();p=P/path
   with PILImage.open(p) as im:w,h=im.size
   story.append(KeepTogether([Image(str(p),width=width,height=width*h/w),para(label,caption)]))
  else:story.append(para(block))
doc=SimpleDocTemplate(str(P/'实验报告.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=44,bottomMargin=44,title='灾前灾后卫星影像建筑损毁分类实验报告',author='DDA4220 项目组')
doc.build(story,onFirstPage=foot,onLaterPages=foot)
print('PDF complete')
