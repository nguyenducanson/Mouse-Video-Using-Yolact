from inference import Mouse_Detection


if __name__ == '__main__':
    mouse = Mouse_Detection()
    video = ""
    flip = False
    print(mouse.inference(video, flip))
