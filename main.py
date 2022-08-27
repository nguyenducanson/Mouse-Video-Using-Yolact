from inference import Mouse_Detection


if __name__ == '__main__':
    mouse = Mouse_Detection()
    video = ""
    print(mouse.inference(video))
