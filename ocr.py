import pytesseract


#Maybe implement splitting image to make first part text and second int
def to_text(image):
    config = ("-l eng --oem 1 --psm 7 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    try:
        data = pytesseract.image_to_data(image, config=config)
        parsed = data.split('\n')
        parsed.pop(0)
        conf = 1.0
        text = ""
        for i in parsed:
            row = i.split("\t")
            # print(row[10])
            if int(row[10]) > 30:
                conf = (conf*float(row[10])*0.01)
                text += row[11]
                text += " "
        out_text = ""

        if conf > 0.55 and len(text) > 4:
            out_text = "Conf: %s | %s" % (conf, text.rstrip(") "))

        return out_text
    except:
        return
