import tkinter as tk
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from model import ConvNet
import matplotlib.pyplot as plt

device = torch.device("cuda")

def predict_digit():
    bbox = image.getbbox()

    if bbox is None:
        result_label.config(text="pred：")
        return

    digit = image.crop(bbox)
    digit.thumbnail((20, 20), Image.Resampling.LANCZOS)

    new_img = Image.new("L", (28, 28), color=0)

    left = (28 - digit.width) // 2
    top = (28 - digit.height) // 2
    new_img.paste(digit, (left, top))

    new_img.save("model_input_28x28.png")

    # show 28*28 image
    # plt.figure(figsize=(5, 5))
    # plt.imshow(new_img, cmap="gray", interpolation="nearest")
    # plt.title("28x28")
    # plt.xticks([0, 10, 20])
    # plt.yticks([0, 10, 20])
    # plt.grid()
    # plt.show()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    img = transform(new_img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.exp(output)
        pred = prob.argmax(dim=1).item()
        confidence = prob[0][pred].item()

    result_label.config(text=f"pred：{pred}，conf：{confidence:.2%}")

def get_x_and_y(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y
def draw(event):
    global last_x, last_y
    canvas.create_line((last_x, last_y, event.x, event.y),fill="white",width=8,smooth=True,capstyle=tk.ROUND)
    draw_image.line((last_x, last_y, event.x, event.y), fill=255, width=8)
    last_x, last_y = event.x, event.y
def clear_canvas():
    global image, draw_image
    canvas.delete('all')
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw_image = ImageDraw.Draw(image)
    result_label.config(text="pred：")


model = ConvNet().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

root = tk.Tk()
root.title('MNIST')
root.configure(background='white')
root.resizable(False, False)
root.geometry('350x350')

canvas_size = 280
frame = tk.Frame(root, background="white")
frame.pack()

image = Image.new("L", (canvas_size, canvas_size), color=0)
draw_image = ImageDraw.Draw(image)

# title_label = tk.Label(frame, text='MNIST')
# title_label.pack()

canvas = tk.Canvas(frame, background='black',width=canvas_size, height=canvas_size)  
canvas.pack()

clear_button = tk.Button(frame, text="Clear" , command=clear_canvas)
clear_button.pack(side=tk.RIGHT)

button = tk.Button(frame, text="Recognition", command=predict_digit)
button.pack(side=tk.RIGHT)

result_label = tk.Label(frame, text="pred：", font=("Arial", 12))
result_label.pack()

canvas.bind("<Button-1>", get_x_and_y)
canvas.bind("<B1-Motion>", draw)
root.mainloop()