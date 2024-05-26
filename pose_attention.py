import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_attention_map(joint_coordinates, confidence_scores, visible,image_size, radius=20, scale=100, variance=1.0):

    epsilon = 1e-6
    attention_map = torch.zeros((1, 1, image_size[0], image_size[1]))
    attention_maps  =[]
    for coords, score, vis in zip(joint_coordinates[0], confidence_scores[0],visible[0]):
        y, x = coords
        if vis <0.4 :
            continue
        if score > 0.7:
            continue
        

        x_range = torch.arange(image_size[1]).float()
        y_range = torch.arange(image_size[0]).float()
        xx, yy = torch.meshgrid(x_range, y_range)
        mask = ((xx - x).pow(2) + (yy - y).pow(2)) <= radius**2


        center_atten = 1 / score**2
        #print(center_atten)
        attention_weight = center_atten* torch.exp(-(torch.pow(xx - x, 2) + torch.pow(yy - y, 2)) / (2.0 * variance + epsilon))



        attention_weight *= scale


        attention_map += attention_weight.view(1, 1, image_size[0], image_size[1])

    attention_map /= attention_map.max()

    return attention_map




def generate_hand_attention_map(joint_coordinates, confidence_scores,image_size, radius=2, scale=100, variance=1.0):

    epsilon = 1e-6

    attention_map = torch.zeros((1, 1, image_size[0], image_size[1]))
    attention_maps  =[]
    if confidence_scores.ndim == 3:
        if not joint_coordinates.shape == confidence_scores.shape:
            confidence_scores = confidence_scores.permute(0,2,1)
    for coords, score in zip(joint_coordinates[0], confidence_scores[0]):
        y, x = coords
        if not isinstance(score,int):
            score = torch.mean(score).item()
        if score > 0.7:
            continue

        x_range = torch.arange(image_size[1]).float()
        y_range = torch.arange(image_size[0]).float()
        xx, yy = torch.meshgrid(x_range, y_range)

        center_atten = 1
        #print(center_atten)
        attention_weight = 10* center_atten* torch.exp(-(torch.pow(xx - x, 2) + torch.pow(yy - y, 2)) / (2.0 * variance + epsilon))

        attention_weight *= scale


        attention_map += attention_weight.view(1, 1, image_size[0], image_size[1])
        #attention_maps.append(attention_weight.view(1, 1, image_size[0], image_size[1]))

    #attention_maps = torch.stack(attention_maps,dim=0)
    #attention_map = torch.max(attention_maps,dim=0)[0]
    attention_map /= attention_map.max()

    return attention_map





def generate_interaction_attention_map(joint_coordinates, confidence_scores, visible,image_size, radius=20, scale=100, variance=1.0):
   
    epsilon = 1e-6

    attention_map = torch.zeros((1, 1, image_size[0], image_size[1]))
    attention_maps  =[]
    for coords, score, vis in zip(joint_coordinates[0], confidence_scores[0],visible[0]):
        y, x = coords
        if vis <0.4 :
            continue
        if score > 0.7:
            continue
        

        x_range = torch.arange(image_size[1]).float()
        y_range = torch.arange(image_size[0]).float()
        xx, yy = torch.meshgrid(x_range, y_range)
        mask = ((xx - x).pow(2) + (yy - y).pow(2)) <= radius**2


        center_atten = 1 / score**2
        #print(center_atten)
        attention_weight = 10* center_atten* torch.exp(-(torch.pow(xx - x, 2) + torch.pow(yy - y, 2)) / (2.0 * variance + epsilon))

        attention_weight *= scale


        attention_map += attention_weight.view(1, 1, image_size[0], image_size[1])

    attention_map /= attention_map.max()

    return attention_map