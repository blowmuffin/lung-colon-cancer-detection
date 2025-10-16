import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf

# ----------------------------
# Load Model
# ----------------------------
loaded_model = tf.keras.models.load_model("D:/lungAndColon_cancer/Model.h5", compile=False)
loaded_model.compile(
    tf.keras.optimizers.Adamax(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

class_labels = ["Lung Adenocarcinoma", "Lung Normal", "Lung SCC", "Colon Adenocarcinoma", "Colon Normal"]

# ----------------------------
# Prediction Function
# ----------------------------
def upload_and_predict():
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return

    # Preprocess
    image = Image.open(file_path)
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = class_labels[tf.argmax(score)]
    confidence = 100 * tf.reduce_max(score)

    # Show Image
    img_display = image.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img_display)
    img_panel.config(image=img_tk)
    img_panel.image = img_tk

    # Update Prediction
    result_label.config(text=f"{predicted_class}\nConfidence: {confidence:.2f}%", fg="#1976D2")

    # Update Probability Table
    for row in prob_table.get_children():
        prob_table.delete(row)
    for i, label in enumerate(class_labels):
        prob_table.insert("", "end", values=(label, f"{score[i]*100:.2f}%"))

# ----------------------------
# Reset Function
# ----------------------------
def reset_app():
    img_panel.config(image='')
    img_panel.image = None
    result_label.config(text="Prediction will appear here", fg="#555")
    for row in prob_table.get_children():
        prob_table.delete(row)

# ----------------------------
# Tkinter Web-like UI
# ----------------------------
root = tk.Tk()
root.title("Lung Cancer Prediction by Ayush")
root.configure(bg="#f5f7fa")
root.state('zoomed')  # Maximize window

# Styling
style = ttk.Style()
style.theme_use("clam")
style.configure("TButton", font=("Segoe UI", 12, "bold"), padding=8, relief="flat", background="#1976D2", foreground="white")
style.map("TButton", background=[("active", "#0D47A1")])
style.configure("TLabel", font=("Segoe UI", 11), background="white", foreground="#333")
style.configure("Treeview.Heading", font=("Segoe UI", 11, "bold"), background="#1976D2", foreground="white")
style.configure("Treeview", font=("Segoe UI", 10), rowheight=25)

# ----------------------------
# Navbar (Header)
# ----------------------------
navbar = tk.Frame(root, bg="#1976D2", height=50)
navbar.pack(fill="x")
nav_title = tk.Label(navbar, text="🩺 Lung & Colon Cancer Detection", font=("Segoe UI", 16, "bold"), bg="#1976D2", fg="white")
nav_title.pack(pady=10)

# ----------------------------
# Scrollable Card
# ----------------------------
canvas = tk.Canvas(root, bg="#f5f7fa")
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas, bg="white", bd=1, relief="solid")

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ----------------------------
# Instructions
# ----------------------------
instructions = tk.Label(
    scrollable_frame,
    text=(
        "Welcome to the Cancer Detection WebApp.\n\n"
        "This tool uses Artificial Intelligence to analyze histopathological images "
        "of lung and colon tissues. By uploading an image, the system predicts whether "
        "the tissue is normal or cancerous, and if cancerous, identifies the type.\n\n"
        "👉 How to use:\n"
        "1. Click 'Upload Image'.\n"
        "2. Select a histopathology image (.jpg/.png).\n"
        "3. Wait for the AI model to analyze.\n"
        "4. View the prediction result and confidence score.\n"
    ),
    font=("Segoe UI", 11),
    bg="white",
    fg="#444",
    justify="left",
    wraplength=850
)
instructions.grid(row=0, column=0, columnspan=3, padx=20, pady=15, sticky="w")

# ----------------------------
# Image Panel
# ----------------------------
img_panel = tk.Label(scrollable_frame, bg="white")
img_panel.grid(row=1, column=0, padx=20, pady=10)

# ----------------------------
# Upload + Reset Buttons
# ----------------------------
upload_button = ttk.Button(scrollable_frame, text="Upload Image", command=upload_and_predict)
upload_button.grid(row=1, column=1, padx=10, pady=5, sticky="n")

reset_button = ttk.Button(scrollable_frame, text="Reset", command=reset_app)
reset_button.grid(row=2, column=1, padx=10, pady=5, sticky="n")

# ----------------------------
# Result Label
# ----------------------------
result_label = tk.Label(scrollable_frame, text="Prediction will appear here", font=("Segoe UI", 13, "bold"), bg="white", fg="#555")
result_label.grid(row=1, column=2, padx=20, pady=10, sticky="n")

# ----------------------------
# Probability Table
# ----------------------------
prob_table = ttk.Treeview(scrollable_frame, columns=("Class", "Probability"), show="headings", height=5)
prob_table.heading("Class", text="Class")
prob_table.heading("Probability", text="Probability")
prob_table.column("Class", anchor="center", width=400)
prob_table.column("Probability", anchor="center", width=150)
prob_table.grid(row=3, column=0, columnspan=3, padx=20, pady=15)

# ----------------------------
# Footer
# ----------------------------
footer = tk.Label(scrollable_frame, text="Group No. 15 | © 2025 Cancer Detection AI", font=("Segoe UI", 9), bg="#f5f7fa", fg="#777")
footer.grid(row=4, column=0, columnspan=3, pady=10)

# ----------------------------
# Run App
# ----------------------------
root.mainloop()
