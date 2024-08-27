from PIL import ImageDraw, ImageFont

# Define the diagram components and their text
components = [
    "Receive Global Model from Central Server",
    "Data Segmentation\n(eg: 80-20 split)",
    "Model Training",
    "Forward Pass",
    "Loss Calculation",
    "Backward Pass",
    "Weight Update",
    "Evaluation\n(On validation set)",
    "Send updates to central server"
]

# Define the size of the image and the spacing between components
component_width = 300
component_height = 100
spacing = 20

# Calculate the width and height of the new image
total_width = component_width * len(components) + spacing * (len(components) + 1)
total_height = component_height + 2 * spacing

# Create a new blank image with white background
new_img = Image.new('RGB', (total_width, total_height), "white")
draw = ImageDraw.Draw(new_img)

# Load a font
font = ImageFont.load_default()

# Draw the components
for i, component in enumerate(components):
    x = spacing + i * (component_width + spacing)
    y = spacing
    draw.rectangle([x, y, x + component_width, y + component_height], outline="black", width=2)
    
    # Get text size and position text in the center of the component
    text_size = draw.textsize(component, font=font)
    text_x = x + (component_width - text_size[0]) / 2
    text_y = y + (component_height - text_size[1]) / 2
    draw.multiline_text((text_x, text_y), component, fill="black", font=font, align="center")

# Save the new image
new_horizontal_img_path = "/mnt/data/horizontal_flow_image.png"
new_img.save(new_horizontal_img_path)

new_horizontal_img_path