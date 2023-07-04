from PIL import ImageDraw,Image

def draw_boxes(image,bounds,color='yellow',width=2):
    image = Image.open(image)
    draw = ImageDraw.Draw(image)
    for bound in bounds:
        p0,p1,p2,p3 = bound[0]
        draw.line([*p0,*p1,*p2,*p3,*p0],fill=color,width=width)
    return image