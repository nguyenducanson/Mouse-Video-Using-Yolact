from inference import Mouse_Detection


if __name__ == '__main__':
    mouse = Mouse_Detection()
    video = "/home/ubuntu/Novodan/Mouse-Video-Using-Yolact/video/#27.mov"
    flip = True
    print(mouse.inference(video, flip))
