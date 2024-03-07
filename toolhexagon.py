from PIL import Image, ImageDraw, ImageFont
from math import pi, sin, cos, sqrt
from scipy.special import comb
import random
import time
import noise 
import numpy as np
import math
import cv2
import psutil
import wmi
import sys

class RandomNumberGenerator:
    """随机的地图块参数"""
    def __init__(self):
        self.generated_numbers = set()

    def generate_number(self):
        number = 10000
        while number in self.generated_numbers:
            number = random.randint(10000, 99999)
        
        self.generated_numbers.add(number)
        return number


class RandomStringGenerator:
    """随机字符串库，用于获取随机名称"""
    def __init__(self, characters):
        self.characters = characters

    def generate_strings(self):
        result = []
        for i in range(3):
            string = ''.join(random.sample(self.characters, i+1))
            result.append(string)
        return random.choice([result[2],result[1]])

    def getstr_list(self,num):
        citys = list()
        for n in range(num):
         citys.append(self.generate_strings())

        return citys


citystr = RandomStringGenerator("北京天津上海重庆广州南昌九江景德镇鹰潭新余萍乡赣州上饶抚州宜春吉安瑞昌共青城庐山乐平瑞金德兴丰城樟树高安井冈山贵溪")
rand = RandomNumberGenerator()

def is_color(point, color, image_data):
    """判断选取该点是否为正确的颜色"""
    x, y = point
    if np.array_equal(image_data[x][y], color):
        return True
    else:
        return False

def poisson_disc_sampling(width, height, min_distance)->list:
    """随机生成min_distance距离的点,并输出点列表"""
    cell_size = min_distance / math.sqrt(2)
    grid_width = int(math.ceil(width / cell_size))
    grid_height = int(math.ceil(height / cell_size))
    grid = [[None for y in range(grid_height)] for x in range(grid_width)]

    def get_grid_coords(point):
        return int(point[0] // cell_size), int(point[1] // cell_size)

    def fits(point):
        gx, gy = get_grid_coords(point)
        if not (0 <= gx < grid_width and 0 <= gy < grid_height):
            return False
        for x in range(max(gx - 2, 0), min(gx + 3, grid_width)):
            for y in range(max(gy - 2, 0), min(gy + 3, grid_height)):
                if grid[x][y] is not None:
                    dx = point[0] - grid[x][y][0]
                    dy = point[1] - grid[x][y][1]
                    if dx * dx + dy * dy < min_distance * min_distance:
                        return False
        return True

    points = []
    active_points = []

    first_point = (int(random.uniform(0,width)), int(random.uniform(0,height)))
    points.append(first_point)
    active_points.append(first_point)
    gx, gy = get_grid_coords(first_point)
    grid[gx][gy] = first_point

    while active_points:
        index = random.randint(0,len(active_points)-1)
        point = active_points[index]
        found_new_point=False
        for i in range(30):
            angle=random.uniform(0.0,(2*math.pi))
            radius=random.uniform(min_distance,(min_distance*2))
            new_x=int(point[0]+radius*math.sin(angle))
            new_y=int(point[1]+radius*math.cos(angle))

            new_point=(new_x,new_y)

            if fits(new_point):
                points.append(new_point)
                active_points.append(new_point)
                gx,gy=get_grid_coords(new_point)
                grid[gx][gy]=new_point
                found_new_point=True
                break

        if not found_new_point:
            active_points.pop(index)

    return points




def hexagon(image_path: str, hexagon_size: int , name:str, list:RandomNumberGenerator)->list:
    """用于绘制六边形地图"""
    image = Image.open(image_path)
    width, height = image.size
    draw = ImageDraw.Draw(image)

    centers = []
    for row in range(height // hexagon_size + 1):
        for col in range(width // hexagon_size + 1):
            x = col * hexagon_size * 3 / 2
            y = row * hexagon_size * sqrt(3) + (col % 2) * hexagon_size * sqrt(3) / 2
            num = list.generate_number()
            draw.text((int(x)-30,int(y)-30),str(num),"black")
            points = [
                (x - hexagon_size / 2, y - hexagon_size * sqrt(3) / 2),
                (x + hexagon_size / 2, y - hexagon_size * sqrt(3) / 2),
                (x + hexagon_size, y),
                (x + hexagon_size / 2, y + hexagon_size * sqrt(3) / 2),
                (x - hexagon_size / 2, y + hexagon_size * sqrt(3) / 2),
                (x - hexagon_size, y)
            ]
            draw.polygon(points,outline="black")
            centers.append({'id':num,'xy':(int(x), int(y))})
    
    image.save(name)
    return centers


def radar_chart(values:list[5]):
   """用于绘制雷达图"""
   size = 400
   bg_color = (255, 255, 255)
   img = Image.new('RGB', (size, size), bg_color)
   draw = ImageDraw.Draw(img)
   center = (size / 2, size / 2)
   radius = size / 3
   N = len(values)
  
   points = []
   for i in range(N):
     angle = i * 2 * pi / N - pi /2 
     x = center[0] + radius * values[i] * cos(angle)
     y = center[1] + radius * values[i] * sin(angle)
     points.append((x,y))
   draw.polygon(points, fill=(200,200,200), outline=(0,0,0))
   
   for i in range(N):
     angle = i *2 * pi / N - pi /2 
     x = center[0] + radius * cos(angle) 
     y = center[1] + radius * sin(angle) 
     draw.line([center,(x,y)], fill=(0,0 ,0)) 
   return img

def paste_image(image1:Image, image2:Image, coordinates:list):
    """将image1绘制在image2中心点上"""
    x, y = coordinates
    width2, height2 = image2.size
    paste_x = x - width2 // 2
    paste_y = y - height2 // 2
    image1.paste(image2, (paste_x, paste_y))
    return image1

def generate_terrain(length:int):
    terrain = ['平原', '山脉', '河流', '海洋']
    result = []
    for i in range(length):
        result.append(random.choice(terrain))
    return result

def draw_text(text:str, image:Image, font_size:int, font_name:str):
    """为图片中间加文字"""
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_name, font_size)
    left, top, right, bottom = draw.textbbox((0, 0), text=text, font=font)
    text_width = right - left
    text_height = bottom - top
    image_width, image_height = image.size
    text_x = (image_width - text_width) // 2
    text_y = (image_height - text_height) // 2
    draw.text((text_x, text_y), text, fill='black', font=font)
    return image

def draw_text_on_image(image:str, points_and_text:dict, offset_xy:list,font_size:int, font_name:str):
    """在偏移值处添加文字"""
    new = Image.open(image)
    font = ImageFont.truetype(font_name, font_size)
    draw = ImageDraw.Draw(new)
    for point, text in points_and_text.items():
        x, y = point
        x += offset_xy[0]
        y += offset_xy[1]
        draw.text((x,y), text,fill='white',font=font)
    
    new.save(image)
    return new


def draw_infantry_badge(width:int, height:int,color:str):
    # 创建一个空白图像
    image = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(image)
    return image

def create_terrainmap(plain,hilly,base):
   # 设置地图大小
    shape = (5000, 5000)
    scale = 1600
    octaves = 25
    persistence = 0.6
    lacunarity = 5

# 生成随机地形数据
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(
                i/scale,
                j/scale,
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=3000,
                repeaty=3000,
                base=base
            )

# 创建彩色图像数据
    color_world = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    print(world)
# 根据地形高度为图像着色
    for i in range(shape[0]):
        for j in range(shape[1]):
            if world[i][j] < -0.05:
                color_world[i][j] = [0, 0, 128] # 深海
            elif world[i][j] < -0.15:
                color_world[i][j] = [65,105,225] # 海洋
            elif world[i][j] < plain:
                color_world[i][j] = [34,139,34] # 平原
            elif world[i][j] < hilly:
                color_world[i][j] = [255,228,196] #高原
            else:
                color_world[i][j] = [139,137,137] # 山峰

    # 创建 PIL Image 对象并显示图像
    img = Image.fromarray(color_world)
    img.save("000.png")

    return img

create_terrainmap(0.1,0.2,100)

def draw_image(img_name:str, points:list, color_code:str, image_path:str,xyoffset:int)->Image:
    """在给定的像素点列表确定颜色绘制图像"""
    name_list = []
    world_img = Image.open(img_name)
    img = Image.open(image_path)
    width, height = img.size

    center_x = width // 2 + xyoffset
    center_y = height // 2 + xyoffset

    color_world = np.array(world_img)

    for point in points:
        x, y = point
        if (color_world[y][x] == color_code).all():
            name_list.append((x,y))
            world_img.paste(img, (x - center_x, y - center_y), img)
        

    world_img.save(img_name)
    return name_list

def check_line_color(image_data, x0, y0, x1, y1, colors):
    """检测两点之间颜色是否为同一种颜色,误差为5"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    err = dx - dy
    different_color_count = 0

    while True:
        if not any(np.array_equal(image_data[x0,y0], color) for color in colors):
            different_color_count += 1
            if different_color_count > 5:
                return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return True


def draw_stripes(image_str:str, points:list,line_width:int):
    image = Image.open(image_str)
    """在地图给定点列表判断颜色，如果为[34,139,34],[255,228,196]则绘制交通线"""
    if len(points) < 2:
        return
    n = len(points)
    adjacency_matrix = [[float('inf')] * n for _ in range(n)]
    
    for i in range(n):
        min_distance = float('inf')
        min_index = -1
        for j in range(n):
            if i == j:
                continue
            x1, y1 = points[i]
            x2, y2 = points[j]
            distance = sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if distance < min_distance:
                min_distance = distance
                min_index = j
        adjacency_matrix[i][min_index] = min_distance
        adjacency_matrix[min_index][i] = min_distance
    
    draw = ImageDraw.Draw(image)
    data = np.array(image)
    for i in range(len(points)):
        for j in range(i+1,len(points)):
            if adjacency_matrix[i][j]!=float('inf'):
                if check_line_color(data,points[i][0],points[i][1],points[j][0],points[j][1],[[34,139,34],[255,228,196]]):
                    draw.line([points[i], points[j]], fill="purple", width=line_width)
    
    image.show()
    image.save(image_str)
    return image


def draw_bezier_curve(points:list, image:list):
    """给定一组控制点 points 和一张图像 image,在图像上绘制贝塞尔曲线并返回图像"""
    num_points = len(points)
    x_points = np.array([p[0] for p in points])
    y_points = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, 1000)

    polynomial_array = np.array([comb(num_points - 1, i) * t**(num_points - 1 - i) * (1 - t)**i for i in range(0,num_points)])

    xvals = np.dot(x_points, polynomial_array)
    yvals = np.dot(y_points, polynomial_array)

    for i in range(len(xvals)-1):
        cv2.line(image,(int(xvals[i]),int(yvals[i])),(int(xvals[i+1]),int(yvals[i+1])),(255,0,0),2)

    return image



def get_system_info_str() ->str:
    if sys.platform == "win32":
        memory = psutil.virtual_memory()
        total_memory = round(memory.total / (1024 ** 3), 2)
        used_memory = round((memory.total - memory.available) / (1024 ** 3), 2)
        cpu = psutil.cpu_percent()

        c = wmi.WMI()
        cpu_str = c.Win32_Processor()[0].Name
        memory = psutil.virtual_memory()
        total_memory = round(memory.total / (1024 ** 3), 2)
        graphics_card = c.Win32_VideoController()[0].Name

        return f"内存占用: {used_memory}GB/{total_memory}GB \n CPU率: {cpu}% \n CPU型号: {cpu_str}  \n 显卡型号: {graphics_card}"
    
    else:
        return None

def generate_circle_image(radius: int):

    # 创建一个带有透明背景的新图像
    image = Image.new("RGBA", (radius * 2, radius * 2), (0, 0, 0, 0))
    
    # 在图像上绘制圆圈
    draw = ImageDraw.Draw(image)
    draw.ellipse((0, 0, radius * 2, radius * 2), fill=(0, 0, 0))
    
    return image


def draw_badge(army_name: str, badge_image: Image, text_list: list, flag_image: Image) ->Image:
    """绘制兵牌"""
    draw = ImageDraw.Draw(badge_image)
    font = ImageFont.truetype("fonts/fzks.ttf", 10)
    text_width = font.getbbox(army_name)[2]
    x = (badge_image.width - text_width) / 2
    draw.text((x, 30), army_name, font=font, fill="black")
    
    font = ImageFont.truetype("fonts/fzks.ttf", 14)
    text = '-'.join(text_list)
    text_width = font.getbbox(text)[2]
    x = (badge_image.width - text_width) / 2
    draw.text((x, 38), text, font=font, fill="black")
    
    badge_image.paste(flag_image, (7, 0))
    return badge_image

"""兵牌定义"""
INFANTRY = "连级兵牌/步兵连.jpg"
ANTI_TANK = "连级兵牌/反坦克连.jpg"
AIR_DEFANCE = "连级兵牌/防空连.jpg"
SAPPER = "连级兵牌/工兵连.jpg"
MECHANIZED_INFANTRY = "连级兵牌/机械化步兵连.jpg"
MECHANIZED_CAVALRY = "连级兵牌/机械化骑兵连.jpg"
AIRBORNE_INFANTRY = "连级兵牌/空降步兵连.jpg"
AIRBORNE_ARTILLERY = "连级兵牌/空降炮兵连.jpg"
ARTILLERY_COMPANY = "连级兵牌/炮兵连.jpg"
CAVALRY_COMPANY = "连级兵牌/骑兵连.jpg"
TRANSPORTATION = "连级兵牌/运输连.jpg"
RECONNAISSANCE = "连级兵牌/侦察连.jpg" 
ARMORED = "连级兵牌/装甲连.jpg"

COLORS = ['red', 'green', 'yellow', 'orange', 'purple', 'pink', 'white', 'gray']
def test():
    army = draw_badge("德 第一军",draw_infantry_badge(60,55,random.choice(COLORS)),["2","3","2"],Image.open(INFANTRY))
    army.show()
    return army 

def draw_map(name:str,big=None,image_type=None) ->dict:
        image = None
        if image_type == "岛":
            image = create_terrainmap(0.09,0.22,0.33,random.randint(0,100),big)
        
        elif image_type == "陆":
            image = create_terrainmap(-0.15,0.2,0.3,random.randint(0,100),big)

        else:
            image = create_terrainmap(-0.09,0.2,0.3,random.randint(0,100),big)

        rand = RandomNumberGenerator()
        picter = hexagon(image,40,rand)
        pointlist = picter['centers']
        image = picter['image']

        citys = poisson_disc_sampling(big-150,big-150,150)

        city_dict = draw_image(image,citys,[34,139,34],generate_circle_image(5),0)
        citys = city_dict['list']
        image = city_dict['image']

        citys_str = citystr.getstr_list(int(len(citys)))
        city_collection = dict(zip(citys,citys_str))
        image = draw_text_on_image(image,city_collection,(3,1),20,"fonts/fzks.ttf")
        return {"point_list":pointlist,"image":image}



def draw_badge_map() ->Image:
    """将兵牌画在地图上"""

import os

def list_files_in_dir(path):
    file_list = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):
            name, ext = os.path.splitext(filename)
            file_list.append(name)
    return file_list
