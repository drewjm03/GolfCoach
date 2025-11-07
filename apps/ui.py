import cv2

class Button:
    def __init__(self, label, x, y, w, h, color_idle=(50,50,50), color_active=(0,160,0)):
        self.label = label
        self.rect = (x, y, w, h)
        self.color_idle = color_idle
        self.color_active = color_active
        self.active = False

    def draw(self, img):
        x,y,w,h = self.rect
        color = self.color_active if self.active else self.color_idle
        cv2.rectangle(img, (x,y), (x+w,y+h), color, -1)
        cv2.putText(img, self.label, (x+10, y+h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def hit(self, px, py):
        x,y,w,h = self.rect
        return (px>=x and px<=x+w and py>=y and py<=y+h)

def draw_ids(img, corners, ids, color=(0,255,255)):
    if ids is None or len(ids) == 0:
        return
    for c, i in zip(corners, ids):
        c4 = c.reshape(4,2).astype(int)
        p = c4.mean(axis=0).astype(int)
        cv2.putText(img, str(int(i[0])), tuple(p), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


