import random
import imutils
import numpy as np
import math 
import cv2

class Face:
    def __init__(self,label,x,y,width,height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.label = label

    def __str__(self):
        return "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(self.label,self.x,self.y,self.width,self.height)

    def rotate_center(self,degree,dims):
        theta = math.radians(degree)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]],dtype=np.float64)
        x_abs, y_abs = (self.x-0.5)*dims[1], (0.5-self.y)*dims[0]
        x_new, y_new = np.dot(rotation_matrix,np.array([[x_abs],[y_abs]]))
        x_new = ((dims[1]/2+x_new)/dims[1])[0]
        y_new = ((dims[0]/2-y_new)/dims[0])[0]
        self.x = x_new
        self.y = y_new

    def flip_horizontal(self):
        self.x = 1 - self.x
    
    def correct_bounding_box(self):
        if self.x - self.width/2 < 0:
            self.width = self.width - (self.width/2 - self.x)
        if self.x + self.width/2 > 1:
            self.width = (1 - self.x)*2
        if self.y - self.height/2 < 0:
            self.height = self.height - (self.height/2 - self.y)
        if self.y + self.height/2 > 1:
            self.height = (1 - self.y)*2
        
    def convert_to_pixels(self,dims):
        x = int(self.x * dims[1])
        y = int(self.y * dims[0])
        width = int(self.width * dims[1])
        height = int(self.height * dims[0])
        return (x,y,width,height)

class Label:
    def __init__(self):
        self.labels = []

    def correct_bounding_box(self):
        for face in self.labels:
            face.correct_bounding_box()

    def extract_from_file(self,path):
        file = open(path, 'r')
        lines = file.readlines()

        for line in lines:
            face = line.split()
            face_label = Face(np.int8(face[0]), np.float16(face[1]), np.float16(face[2]), np.float16(face[3]), np.float16(face[4]))
            self.labels.append(face_label)
        return self.labels

    def add(self,face):
        self.labels.append(face)
    
    def __iter__(self):
        for face in self.labels:
            yield face

    def __getitem__(self,index):
        return self.labels[index]

    def save(self,dir):
        file = open(dir, "w")
        for face in self.labels:
            file.write(str(face) + "\n")
        file.close()


class FaceAugment:
    def calculate_crop_ratio(self,ratio):
        bins = len(self.cdf)
        index = int(ratio/(max(self.cdf)/bins)-1)
        if index >= bins:
            return -1
        return self.cdf[index]

    def get_rotate_degree(self,n=1,start=-15,end=15):
        rotate_degrees = []
        for i in range(n):
            degree = (end-start)/n*i - end
            
            degree += random.uniform(0,(end-start)/n)
            rotate_degrees.append(degree)
        if n == 1:
            return rotate_degrees[0]
        return rotate_degrees

    def rotate_image(self,image,label,degree):
        rotated = imutils.rotate(image, angle=degree)
        for face in label:
            face.rotate_center(degree,rotated.shape[:2])
        return rotated, label

    def calculate_new_dimensions(self,face,old_dims):
        new_ratio = random.uniform(3,3.5)
        new_w = int(old_dims[1] / math.sqrt(new_ratio))
        new_h = int(old_dims[0] / math.sqrt(new_ratio))
        return new_w, new_h

    def get_random_crop_position(self,face,old_dims,new_width,new_height):
        face_pixel = face.convert_to_pixels(old_dims)
        x_crop = face_pixel[0] - new_width/2
        y_crop = face_pixel[1] - new_height/2

        w_range = new_width/2 - face_pixel[2]/2
        h_range = new_height/2 - face_pixel[3]/2

        x_crop += random.randint(- int(w_range), int(w_range))
        y_crop += random.randint(- int(h_range), int(h_range))
        
        x_crop = max(x_crop, 0)
        y_crop = max(y_crop, 0)
        x_crop = min(x_crop, old_dims[1]-new_width)
        y_crop = min(y_crop, old_dims[0]-new_height)
        
        return [np.ceil(x_crop), np.ceil(y_crop)]
    
    def get_intersect_percent(self,crop_pos, face_pos):
        crop_top_left_x = crop_pos[0] - crop_pos[2]/2
        crop_top_left_y = crop_pos[1] - crop_pos[3]/2
        crop_bottom_right_x = crop_pos[0] + crop_pos[2]/2
        crop_bottom_right_y = crop_pos[1] + crop_pos[3]/2
        face_top_left_x = face_pos[0] - face_pos[2]/2
        face_top_left_y = face_pos[1] - face_pos[3]/2
        face_bottom_right_x = face_pos[0] + face_pos[2]/2
        face_bottom_right_y = face_pos[1] + face_pos[3]/2

        if crop_top_left_x > face_bottom_right_x or crop_bottom_right_x < face_top_left_x:
            return 0

        if crop_top_left_y > face_bottom_right_y or crop_bottom_right_y < face_top_left_y:
            return 0
        
        if crop_top_left_x < face_top_left_x and crop_top_left_y < face_top_left_y:
            return 1

        delta_x = min(np.abs(crop_top_left_x - face_bottom_right_x), np.abs(crop_bottom_right_x - face_top_left_x))
        delta_y = min(np.abs(crop_top_left_y - face_bottom_right_y), np.abs(crop_bottom_right_y - face_top_left_y))

        intersection_area = delta_x * delta_y

        face_area = face_pos[2]*face_pos[3]
        crop_area = crop_pos[2]*crop_pos[3]

        percent = intersection_area / (face_area  + crop_area)

        return percent

    def get_face_intersect(self,crop_pos,new_w,new_h,faces,org_dims):
        collisions = []
        avoids = []

        for face in faces:
            box_1 = (crop_pos[0]+new_w/2, crop_pos[1]+new_h/2,new_w,new_h)
            box_2 = face.convert_to_pixels(org_dims)
            intersection = self.get_intersect_percent(box_1,box_2)
            # print(face.label,intersection)
            if 0.3 <= intersection < 1:
                collisions.append(face)
            if 0 < intersection < 0.3:
                avoids.append(face)
            
        return collisions, avoids

    def cover_collision(self,crop_pos,face_pos):
        crop_top_left_x = crop_pos[0] - crop_pos[2]/2
        crop_top_left_y = crop_pos[1] - crop_pos[3]/2
        crop_bottom_right_x = crop_pos[0] + crop_pos[2]/2
        crop_bottom_right_y = crop_pos[1] + crop_pos[3]/2

        face_top_left_x = face_pos[0] - face_pos[2]/2
        face_top_left_y = face_pos[1] - face_pos[3]/2
        face_bottom_right_x = face_pos[0] + face_pos[2]/2
        face_bottom_right_y = face_pos[1] + face_pos[3]/2


        delta_x = face_pos[2] - min(np.abs(crop_top_left_x - face_bottom_right_x), np.abs(crop_bottom_right_x - face_top_left_x))
        delta_y = face_pos[3] - min(np.abs(crop_top_left_y - face_bottom_right_y), np.abs(crop_bottom_right_y - face_top_left_y))

        delta_x = delta_x + random.uniform(0,delta_x*0.5)
        delta_y = delta_y + random.uniform(0,delta_y*0.5)

        if crop_pos[0] > face_pos[0]:
            x_crop = crop_pos[0] - delta_x
        else:
            x_crop = crop_pos[0] + delta_x
        if crop_pos[1] > face_pos[1]:
            y_crop = crop_pos[1] - delta_y
        else:
            y_crop = crop_pos[1] + delta_y
        return x_crop, y_crop

    def avoid_collision(self,crop_pos,face_pos):
        crop_top_left_x = crop_pos[0] - crop_pos[2]/2
        crop_top_left_y = crop_pos[1] - crop_pos[3]/2
        crop_bottom_right_x = crop_pos[0] + crop_pos[2]/2
        crop_bottom_right_y = crop_pos[1] + crop_pos[3]/2

        face_top_left_x = face_pos[0] - face_pos[2]/2
        face_top_left_y = face_pos[1] - face_pos[3]/2
        face_bottom_right_x = face_pos[0] + face_pos[2]/2
        face_bottom_right_y = face_pos[1] + face_pos[3]/2


        delta_x = min(np.abs(crop_top_left_x - face_bottom_right_x), np.abs(crop_bottom_right_x - face_top_left_x))
        delta_y = min(np.abs(crop_top_left_y - face_bottom_right_y), np.abs(crop_bottom_right_y - face_top_left_y))

        # print("Avoid", delta_x,delta_y)
        delta_x = delta_x + random.uniform(0,delta_x*0.5)
        delta_y = delta_y + random.uniform(0,delta_y*0.5)

        x_crop, y_crop = crop_pos[0], crop_pos[1]
        if delta_x < delta_y:
            if crop_pos[0] > face_pos[0]:
                x_crop = crop_pos[0] + delta_x
            else:
                x_crop = crop_pos[0] - delta_x
        else:
            if crop_pos[1] > face_pos[1]:
                y_crop = crop_pos[1] + delta_y
            else:
                y_crop = crop_pos[1] - delta_y
        return x_crop, y_crop

    def find_crop_position(self,crop,collisions,avoids,old_dims):
        x_crop, y_crop, new_w, new_h = crop[0], crop[1], crop[2], crop[3]
        
        for avoid in avoids :
            x_crop, y_crop = self.avoid_collision((x_crop, y_crop, new_w, new_h), avoid.convert_to_pixels(old_dims))

        for collision in collisions:
            x_crop, y_crop = self.cover_collision((x_crop, y_crop, new_w, new_h), collision.convert_to_pixels(old_dims))

        return (x_crop, y_crop, new_w, new_h)

    def calculate_new_labels(self,label,crop,old_dims):
        new_label = Label()
        for face in label:
            face_pos = face.convert_to_pixels(old_dims)
            if (crop[0] < face_pos[0] < crop[0] + crop[2]) and (crop[1] < face_pos[1] < crop[1] + crop[3]):
                new_face = Face(face.label,(face_pos[0] - crop[0])/crop[2], (face_pos[1]-crop[1])/crop[3], face_pos[2]/crop[2], face_pos[3]/crop[3])
                new_label.add(new_face)
        return new_label

    def flip_image(self,image, label):
        flipped_image = cv2.flip(image,1)
        for face in label:
            face.flip_horizontal()
        return flipped_image, label

    def augment_face(self,image,label,face_index):
        original_dims = image.shape[:2]
        degree = self.get_rotate_degree()
        rotated, label = self.rotate_image(image,label,degree)

        new_w, new_h = self.calculate_new_dimensions(label[face_index],original_dims)
        
        crop_pos = self.get_random_crop_position(label[face_index],original_dims,new_w,new_h)

        crop = (crop_pos[0] + new_w/2, crop_pos[1] + new_h/2, new_w, new_h)

        collisions, avoids = self.get_face_intersect(crop_pos,new_w,new_h,label,original_dims)

        x_crop, y_crop, new_w, new_h = self.find_crop_position(crop,collisions,avoids,original_dims)

        x_crop = max(x_crop, 0)
        y_crop = max(y_crop, 0)
        x_crop = min(x_crop, original_dims[1]-new_w)
        y_crop = min(y_crop, original_dims[0]-new_h)

        x_crop = int(x_crop - new_w/2)
        y_crop = int(y_crop - new_h/2)


        new_label = self.calculate_new_labels(label,(x_crop,y_crop,new_w,new_h),original_dims)

        random_toss = np.random.uniform(0, 1)
        
        if random_toss >= 0.5:
            return self.flip_image(rotated[y_crop:y_crop+new_h,x_crop:x_crop+new_w,::],new_label) 

        return rotated[y_crop:y_crop+new_h,x_crop:x_crop+new_w,::], new_label

