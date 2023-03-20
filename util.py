import base64

def img_to_bytes(img_path):
    with open(img_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    return b64_string