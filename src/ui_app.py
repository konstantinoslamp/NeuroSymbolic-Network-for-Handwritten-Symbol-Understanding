import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
import sys
import os
# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from bridge.neurosymbolic_connector import NeurosymbolicSolver

class DrawingApp:
    """Simple UI for drawing digits and operators"""
    
    def __init__(self, root, model_path='src/neural/trained_cnn_model.pkl'):
        self.root = root
        self.root.title("Handwritten Arithmetic Solver")
        
        # Canvas size
        self.canvas_width = 560  # 28*20 for better drawing
        self.canvas_height = 280  # 28*10
        
        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(padx=10, pady=10)
        
        # Title
        title = Label(main_frame, text="Draw an arithmetic expression (e.g., 3+5)", 
                     font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Canvas for drawing
        self.canvas = Canvas(main_frame, width=self.canvas_width, height=self.canvas_height,
                            bg='white', cursor='cross')
        self.canvas.pack()
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # PIL Image for processing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)  # White background
        self.draw_image = ImageDraw.Draw(self.image)
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        # Buttons
        self.clear_btn = Button(button_frame, text="Clear", command=self.clear_canvas,
                               font=("Arial", 12), width=10, bg="#ff6b6b", fg="white")
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.recognize_btn = Button(button_frame, text="Recognize & Solve", 
                                   command=self.recognize_and_solve,
                                   font=("Arial", 12), width=15, bg="#4CAF50", fg="white")
        self.recognize_btn.pack(side=tk.LEFT, padx=5)
        
        # Result display
        result_frame = tk.Frame(main_frame)
        result_frame.pack(pady=10)
        
        self.result_label = Label(result_frame, text="Draw an expression and click 'Recognize & Solve'",
                                 font=("Arial", 14), fg="blue", wraplength=500)
        self.result_label.pack()
        
        # Variables for drawing
        self.last_x = None
        self.last_y = None
        self.brush_width = 15

        # Load neurosymbolic solver
        print("Initializing Neurosymbolic Solver...")
        try:
            self.solver = NeurosymbolicSolver(model_path)
            print("✓ Solver loaded successfully!")
        except FileNotFoundError:
            print(f"⚠ Warning: Model file '{model_path}' not found.")
            print("  Please train the model first: python src/neural/train.py")
            self.solver = None
        
    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        """Draw on canvas"""
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                   width=self.brush_width, fill='black',
                                   capstyle=tk.ROUND, smooth=tk.TRUE)
            
            # Draw on PIL image
            self.draw_image.line([self.last_x, self.last_y, event.x, event.y],
                               fill=0, width=self.brush_width)
            
        self.last_x = event.x
        self.last_y = event.y
    
    def clear_canvas(self):
        """Clear the canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 255)
        self.draw_image = ImageDraw.Draw(self.image)
        self.result_label.config(text="Canvas cleared. Draw a new expression!")
        self.last_x = None
        self.last_y = None
    
    def segment_characters(self):
        """
        Segment the drawn image into individual characters.
        Returns list of (x_pos, image_array) tuples.

        Uses vertical projection with gap-closing: short gaps (< 20px wide)
        inside a character region are bridged so that '+' strokes don't split.
        If projection-based segmentation finds != 3 segments, falls back to
        dividing the canvas into 3 equal columns.
        """
        img_array = np.array(self.image)

        # Vertical projection: count dark pixels per column
        vertical_projection = np.sum(img_array < 200, axis=0)
        is_gap = vertical_projection < 3  # column has fewer than 3 dark pixels

        # Close short gaps (bridges internal gaps in '+', '-', etc.)
        GAP_CLOSE = 20  # px — gaps narrower than this inside a character are filled
        closed_gap = is_gap.copy()
        i = 0
        while i < len(closed_gap):
            if closed_gap[i]:  # start of a gap
                j = i
                while j < len(closed_gap) and closed_gap[j]:
                    j += 1
                gap_width = j - i
                # Only close if surrounded by ink on both sides
                left_ink  = i > 0 and not is_gap[i - 1]
                right_ink = j < len(closed_gap) and not is_gap[j]
                if gap_width < GAP_CLOSE and left_ink and right_ink:
                    closed_gap[i:j] = False
                i = j
            else:
                i += 1

        # Find character boundaries from closed projection
        raw_segments = []
        in_char = False
        start = 0
        for i, is_g in enumerate(closed_gap):
            if not is_g and not in_char:
                start = i
                in_char = True
            elif is_g and in_char:
                if i - start > 10:
                    raw_segments.append((start, i))
                in_char = False
        if in_char:
            raw_segments.append((start, len(closed_gap)))

        # Merge any segments that are very close together (< 15px gap)
        segments = []
        for seg in raw_segments:
            if segments and seg[0] - segments[-1][1] < 15:
                segments[-1] = (segments[-1][0], seg[1])
            else:
                segments.append(list(seg))

        print(f"  Projection segmentation found {len(segments)} segment(s): {segments}")

        # Fallback: split canvas into 3 equal thirds
        if len(segments) != 3:
            print(f"  Falling back to equal-thirds split (canvas width={self.canvas_width})")
            w = self.canvas_width
            segments = [
                [0,          w // 3],
                [w // 3,     2 * w // 3],
                [2 * w // 3, w],
            ]

        # Extract and resize each segment to 28x28
        characters = []
        for start_x, end_x in segments:
            char_img = img_array[:, start_x:end_x]

            # Crop vertical whitespace
            vertical_proj = np.sum(char_img < 200, axis=1)
            nonzero_rows = np.where(vertical_proj > 0)[0]
            if len(nonzero_rows) == 0:
                # Blank region — make an empty 28x28
                characters.append((start_x, np.zeros((28, 28), dtype=np.float32)))
                continue

            top    = max(0, nonzero_rows[0] - 5)
            bottom = min(char_img.shape[0], nonzero_rows[-1] + 5)
            char_img = char_img[top:bottom, :]

            # Pad to square then resize to 28x28
            char_pil = Image.fromarray(char_img)
            w, h = char_pil.size
            max_dim = max(w, h, 1)
            square_img = Image.new('L', (max_dim, max_dim), 255)
            square_img.paste(char_pil, ((max_dim - w) // 2, (max_dim - h) // 2))
            resized = square_img.resize((28, 28), Image.Resampling.LANCZOS)

            char_array = np.array(resized).astype(np.float32) / 255.0
            char_array = 1.0 - char_array  # invert: MNIST is white-on-black

            characters.append((start_x, char_array))

        return characters
    
    def recognize_and_solve(self):
        """Segment drawing, recognize symbols, and solve expression"""
        if self.solver is None:
            self.result_label.config(
                text="⚠ Model not loaded! Please train first: python src/neural/train.py",
                fg='red'
            )
            return
            
        print("\n" + "="*70)
        print("STARTING RECOGNITION")
        print("="*70)
        
        # Segment the image
        segments = self.segment_characters()
        print(f"\n  Using {len(segments)} segments")
        
        # Extract just the images (not positions)
        images = [seg[1] for seg in segments]
        
        # Use neurosymbolic solver
        result = self.solver.solve_expression(images)
        
        # Display result
        if result['success']:
            expr = result['expression']
            ans = result['result']
            result_text = f"✓ {expr} = {ans}"
            self.result_label.config(text=result_text, fg='green', font=('Arial', 16, 'bold'))
            print(f"\n{'='*70}")
            print(f"FINAL ANSWER: {result_text}")
            print(f"{'='*70}\n")
        else:
            result_text = f"✗ {result['explanation']}"
            self.result_label.config(text=result_text, fg='red')
        
        # Show validation messages if any
        if result.get('validations'):
            print("\nValidation details:")
            for msg in result['validations']:
                print(f"  {msg}")

def main():
    """Run the application"""
    import os
    
    # Check if model exists
    model_path = 'src/neural/trained_cnn_model.pkl'
    if not os.path.exists(model_path):
        print("="*70)
        print("⚠ WARNING: Model file not found!")
        print("="*70)
        print(f"Looking for: {model_path}")
        print("\nTo train the model, run:")
        print("  python src/neural/train.py")
        print("\nThe UI will still open, but recognition won't work until trained.")
        print("="*70 + "\n")
    
    root = tk.Tk()
    app = DrawingApp(root, model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
