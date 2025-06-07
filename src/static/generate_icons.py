from PIL import Image, ImageDraw, ImageFont
import os

def create_icon(size, output_path, maskable=False):
    # Create a new image with a white background
    img = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(img)
    
    # Calculate padding for maskable icons
    padding = size // 8 if maskable else 0
    
    # Draw gradient background
    for y in range(size):
        for x in range(size):
            # Create a gradient from top-left to bottom-right
            r = int(74 + (52-74) * (x/size))  # From #4a to #34
            g = int(152 + (206-152) * (x/size))  # From #98 to #ce
            b = int(219 + (113-219) * (x/size))  # From #db to #71
            draw.point((x, y), fill=(r, g, b))
    
    # Add text
    try:
        # Try to load a system font
        font_size = size // 3
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw text
    text = "AI"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    x = (size - text_width) // 2
    y = (size - text_height) // 2
    draw.text((x, y), text, fill='white', font=font)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, 'PNG')

def main():
    # Generate regular icons
    create_icon(192, 'src/static/icons/icon-192.png')
    create_icon(512, 'src/static/icons/icon-512.png')
    
    # Generate iOS specific icons
    create_icon(152, 'src/static/icons/icon-152.png')  # iPad
    create_icon(167, 'src/static/icons/icon-167.png')  # iPad Pro
    create_icon(180, 'src/static/icons/icon-180.png')  # iPhone
    
    # Generate maskable icons
    create_icon(192, 'src/static/icons/icon-192-maskable.png', maskable=True)

if __name__ == '__main__':
    main() 