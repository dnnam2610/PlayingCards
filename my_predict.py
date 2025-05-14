from ultralytics import YOLO
import cv2
import numpy as np
from itertools import combinations
from datetime import datetime
import argparse

video_writer = None

def get_parse():
    parser = argparse.ArgumentParser(description="Poker game")
    parser.add_argument("--input-type", "-i", type=str,required=True, help="0:cam/image/videos path")
    parser.add_argument("--model", "-m", default="best.pt", type=str, help="Path to model")
    parser.add_argument("--save", "-s", default=True, type=bool, help="Save model")
    args = parser.parse_args()
    return args

def convert_input_type(args):
    out = None
    if args.input_type == "0":
        out = 0
    else:
        out = args.input_type  # This could be a path to image or video
    return out


# Function to evaluate poker hand rank
def evaluate_hand(cards_suit, cards_rank):
    hand = None
    colour = None
    rank = None

    if len(set(cards_suit)) == 1:
        if max(cards_rank) - min(cards_rank) == 4:
            if max(cards_rank) == 14:
                hand = "!!!! ROYAL FLUSHHHHH !!!!"
                colour = (0, 0, 255)
                rank = 1
            else:
                hand = "STRAIGHT FLUSH !!!!"
                colour = (255, 102, 0)
                rank = 2
        elif max(cards_rank) - np.sum(sorted(cards_rank)[:-1]) == 0:
            hand = "STRAIGHT"
            colour = (0, 255, 255)
            rank = 6
        else:
            hand = "FLUSH"
            colour = (0, 255, 0)
            rank = 5

    elif len(set(cards_rank)) == 2:
        _, counts_elements = np.unique(cards_rank, return_counts=True)
        if 4 in counts_elements:
            hand = "FOUR OF A KIND!!"
            colour = (255, 255, 0)
            rank = 3
        else:
            hand = "FULL HOUSE !!"
            colour = (255, 153, 51)
            rank = 4

    elif len(set(cards_rank)) == 5 and (max(cards_rank) - min(cards_rank) == 4 or (max(cards_rank) - np.sum(sorted(cards_rank)[:-1]) == 0)):
        hand = "STRAIGHT"
        colour = (60, 255, 120)
        rank = 6

    elif len(set(cards_rank)) == 3:
        _, counts_elements = np.unique(cards_rank, return_counts=True)
        if 3 in counts_elements:
            hand = "Three of a kind"
            colour = (204, 102, 255)
            rank = 7
        else:
            hand = "2 Pairs"
            colour = (204, 204, 0)
            rank = 8

    elif len(set(cards_rank)) == 4:
        hand = "Pair"
        colour = (0, 0, 0)
        rank = 9

    else:
        hand = "High card =(("
        colour = (0, 0, 0)
        rank = 10

    return rank, hand, colour

# Function to draw a bounding box and label on frame
def draw_card_box(frame, label, confidence, xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    text = f"{label} ({confidence:.2f})"
    suit = label[-1]

    # Choose color based on card suit
    if suit in ['H', 'D']:
        color = (0, 0, 255)  # Red
    elif suit in ['S', 'C']:
        color = (0, 0, 0)    # Black
    else:
        color = (128, 128, 128)  # Unknown suit

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Function to extract cards from detected boxes
def extract_detected_cards(boxes, model, frame):
    cards = []
    for box in boxes:
        xyxy = box.xyxy[0]
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        draw_card_box(frame, label, conf, xyxy)
        cards.append(label)
    return list(set(cards))  # Remove duplicates

# Function to analyze current hand and return result
def analyze_hand(cards):
    num_cards = len(cards)
    highest_rank = 11
    best_hand = None
    best_color = None

    if num_cards == 2:
        suits = [c[-1] for c in cards]
        ranks = convert_card_ranks(cards)
        if len(set(ranks)) == 1:
            best_hand = "Pair"
            best_color = (0, 0, 0)
        else:
            best_hand = "High card =(("
            best_color = (0, 0, 0)

    elif num_cards in range(5, 8):
        for combo in combinations(cards, 5):
            suits = [c[-1] for c in combo]
            ranks = convert_card_ranks(combo)
            rank, hand, color = evaluate_hand(suits, ranks)
            if rank < highest_rank:
                highest_rank = rank
                best_hand = hand
                best_color = color

    return best_hand, best_color, highest_rank

# Convert face card strings to numeric ranks
def convert_card_ranks(cards):
    return [
        11 if c[:-1] == "J" else
        12 if c[:-1] == "Q" else
        13 if c[:-1] == "K" else
        14 if c[:-1] == "A" else
        int(c[:-1])
        for c in cards
    ]

# Draw the evaluated hand at the top of the frame
def display_hand_result(frame, hand_text, color):
    if not hand_text or not color:
        return
    _, img_width = frame.shape[:2]
    (text_width, _), _ = cv2.getTextSize(hand_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    x = (img_width - text_width) // 2
    y = 50
    cv2.putText(frame, hand_text, (x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=color,
                thickness=2,
                lineType=cv2.LINE_AA)

# Main frame processing function
def process_frame(result, model, save_enabled=True):
    frame = result.orig_img
    boxes = result.boxes

    detected_cards = extract_detected_cards(boxes, model, frame)
    hand, color, _ = analyze_hand(detected_cards)
    display_hand_result(frame, hand, color)
    if save_enabled:
        save_video(frame)
    return frame

def save_video(frame, output_path="output.mp4", fps=30):
    global video_writer
    height, width = frame.shape[:2]

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    video_writer.write(frame)


if __name__ == "__main__":
    args = get_parse()
    model = YOLO(args.model)
    for result in model.predict(source=convert_input_type(args), stream=True, show=True):
        frame = process_frame(result, model, args.save)
        cv2.imshow("YOLOv8 Poker Hand Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
