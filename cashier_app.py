# Barcode Cashier System
# Kode dibuat oleh: Afra Fadhma Dinata
# Tgl pembuatan: 16/4/2025

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime

# =========== KONSTANTA & KONFIGURASI ===========
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
MODEL_PATH = "barcode_model.h5"
PRODUCT_DB = "product_database.json"

class BarcodeRecognitionSystem:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.product_db = {}
        self.load_product_database()
        
    def load_product_database(self):
        # Coba load database produk atau buat yg baru kalo gada
        try:
            if os.path.exists(PRODUCT_DB):
                with open(PRODUCT_DB, 'r') as f:
                    self.product_db = json.load(f)
            else:
                # Contoh database dummy buat testing
                self.product_db = {
                    "123456789012": {"name": "Coca Cola 1.5L", "price": 15000},
                    "789123456789": {"name": "Indomie Goreng", "price": 3500},
                    "456789123456": {"name": "Pocari Sweat 500ml", "price": 7000},
                    "321654987012": {"name": "Oreo Original 133g", "price": 9500},
                    "654321987654": {"name": "Chitato 75g", "price": 12000}
                }
                self.save_product_database()
        except Exception as e:
            print(f"Error loading database: {e}")
            self.product_db = {}
    
    def save_product_database(self):
        # Simpan database produk ke file
        try:
            with open(PRODUCT_DB, 'w') as f:
                json.dump(self.product_db, f, indent=4)
        except Exception as e:
            print(f"Error saving database: {e}")
    
    def add_product(self, barcode, name, price):
        # Tambah produk ke database
        self.product_db[barcode] = {"name": name, "price": float(price)}
        self.save_product_database()
    
    def get_product(self, barcode):
        # Ambil info produk dari database
        return self.product_db.get(barcode, None)
    
    def create_dataset_structure(self, base_path="dataset"):
        # Bikin struktur folder untuk dataset barcode
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        # Bikin subfolder untuk tiap kategori barcode
        for barcode in self.product_db.keys():
            folder_path = os.path.join(base_path, barcode)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                
        return base_path
    
    def prepare_model(self):
        # Bikin model atau load model yang udah ada
        if os.path.exists(MODEL_PATH):
            print("Loading existing model...")
            self.model = load_model(MODEL_PATH)
            return True
            
        print("Creating new model...")
        try:
            # Check folder dataset
            if not os.path.exists("dataset"):
                print("Dataset folder not found! Creating example structure...")
                self.create_dataset_structure()
                return False
                
            # Data augmentation buat nambah variasi data
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                shear_range=0.2,
                fill_mode='nearest'
            )
            
            # Load data training
            train_data = train_datagen.flow_from_directory(
                'dataset',
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='training'
            )
            
            # Load data validasi
            val_data = train_datagen.flow_from_directory(
                'dataset',
                target_size=IMG_SIZE,
                batch_size=BATCH_SIZE,
                class_mode='categorical',
                subset='validation'
            )
            
            # Simpan nama kelas (barcode)
            self.class_names = list(train_data.class_indices.keys())
            
            # Transfer learning dengan MobileNetV2
            base_model = MobileNetV2(
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze sebagian layer base model
            for layer in base_model.layers[:-20]:
                layer.trainable = False
                
            # Buat model lengkap
            self.model = Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(len(self.class_names), activation='softmax')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks untuk improved training
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
            ]
            
            # Train the model
            history = self.model.fit(
                train_data,
                epochs=EPOCHS,
                validation_data=val_data,
                callbacks=callbacks
            )
            
            # Simpan model
            self.model.save(MODEL_PATH)
            
            # Plot history training
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_accuracy'], label='validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig('training_history.png')
            
            return True
            
        except Exception as e:
            print(f"Error preparing model: {e}")
            return False
    
    def recognize_barcode(self, image_path):
        # Recognize barcode from image
        try:
            if self.model is None:
                if not self.prepare_model():
                    return None
                    
            # Load dan preprocess gambar
            img = tf.keras.preprocessing.image.load_img(
                image_path, target_size=IMG_SIZE
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            predictions = self.model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Cek confidence > threshold
            if confidence > 0.7:  # thresold 70%
                barcode = self.class_names[predicted_class]
                return barcode
            else:
                return None
                
        except Exception as e:
            print(f"Error recognizing barcode: {e}")
            return None

class CashierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Barcode Cashier System - by Afra Fadhma Dinata")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.barcode_system = BarcodeRecognitionSystem()
        self.cart = []  # Simpan items di keranjang
        self.selected_image = None
        
        self.create_gui()
        
    def create_gui(self):
        # Buat frame utama
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(header_frame, text="Barcode Cashier System", font=("Arial", 18, "bold")).pack(side=tk.LEFT, padx=10)
        ttk.Label(header_frame, text="by Afra Fadhma Dinata", font=("Arial", 10, "italic")).pack(side=tk.RIGHT, padx=10)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Barcode frame (kiri)
        barcode_frame = ttk.LabelFrame(content_frame, text="Barcode Scanner", padding=10)
        barcode_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Preview image
        self.image_preview = ttk.Label(barcode_frame)
        self.image_preview.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Button frame
        button_frame = ttk.Frame(barcode_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Browse Image", command=self.browse_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Scan Barcode", command=self.scan_barcode).pack(side=tk.LEFT, padx=5)
        
        # Output frame
        output_frame = ttk.Frame(barcode_frame)
        output_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(output_frame, text="Barcode:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.barcode_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.barcode_var, width=30).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(output_frame, text="Product:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.product_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.product_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(output_frame, text="Price:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.price_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.price_var, width=30).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(output_frame, text="Quantity:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.qty_var = tk.StringVar(value="1")
        ttk.Entry(output_frame, textvariable=self.qty_var, width=30).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        ttk.Button(output_frame, text="Add to Cart", command=self.add_to_cart).grid(row=4, column=0, columnspan=2, pady=10)
        
        # Cart frame (kanan)
        cart_frame = ttk.LabelFrame(content_frame, text="Shopping Cart", padding=10)
        cart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Cart table
        cart_table_frame = ttk.Frame(cart_frame)
        cart_table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(cart_table_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview untuk cart
        self.cart_table = ttk.Treeview(
            cart_table_frame,
            columns=("product", "price", "qty", "subtotal"),
            show="headings",
            yscrollcommand=scrollbar.set
        )
        
        self.cart_table.heading("product", text="Product")
        self.cart_table.heading("price", text="Price")
        self.cart_table.heading("qty", text="Qty")
        self.cart_table.heading("subtotal", text="Subtotal")
        
        self.cart_table.column("product", width=150)
        self.cart_table.column("price", width=70)
        self.cart_table.column("qty", width=50)
        self.cart_table.column("subtotal", width=80)
        
        self.cart_table.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.cart_table.yview)
        
        # Delete button
        ttk.Button(cart_frame, text="Remove Item", command=self.remove_item).pack(fill=tk.X, pady=(5, 10))
        
        # Total frame
        total_frame = ttk.Frame(cart_frame)
        total_frame.pack(fill=tk.X)
        
        ttk.Label(total_frame, text="Total Amount:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, pady=5)
        self.total_var = tk.StringVar(value="Rp 0")
        ttk.Label(total_frame, textvariable=self.total_var, font=("Arial", 12, "bold")).pack(side=tk.RIGHT, pady=5)
        
        # Checkout button
        ttk.Button(cart_frame, text="Checkout", command=self.checkout).pack(fill=tk.X, pady=10)
        
        # Add some dummy data to products if needed
        if not os.path.exists(PRODUCT_DB):
            self.add_dummy_products()
            
        # Check model
        if not os.path.exists(MODEL_PATH):
            messagebox.showinfo("Model Info", "Model not found. You need to train a model first or add images to dataset.")
        
    def add_dummy_products(self):
        # Add dummy barcode database
        print("Creating dummy product database...")
    
    def browse_image(self):
        # Open file dialog untuk select image
        file_path = filedialog.askopenfilename(
            title="Select Barcode Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.selected_image = file_path
            # Tampilkan preview image
            img = Image.open(file_path)
            img = img.resize((300, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            self.image_preview.configure(image=img_tk)
            self.image_preview.image = img_tk  # Keep reference
            
    def scan_barcode(self):
        # Scan barcode dari image
        if not self.selected_image:
            messagebox.showerror("Error", "Please select an image first!")
            return
            
        # Loading indikator
        self.root.config(cursor="wait")
        self.root.update()
        
        try:
            # Recognize barcode
            barcode = self.barcode_system.recognize_barcode(self.selected_image)
            
            if barcode:
                self.barcode_var.set(barcode)
                
                # Lookup product info
                product_info = self.barcode_system.get_product(barcode)
                if product_info:
                    self.product_var.set(product_info["name"])
                    self.price_var.set(str(product_info["price"]))
                else:
                    self.product_var.set("Unknown Product")
                    self.price_var.set("0")
                    messagebox.showwarning("Warning", f"Barcode {barcode} not found in database!")
            else:
                messagebox.showerror("Error", "Cannot recognize barcode from image!")
                self.barcode_var.set("")
                self.product_var.set("")
                self.price_var.set("")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error scanning barcode: {e}")
            
        finally:
            self.root.config(cursor="")
            
    def add_to_cart(self):
        # Add item to cart
        barcode = self.barcode_var.get()
        product = self.product_var.get()
        
        try:
            price = float(self.price_var.get())
            qty = int(self.qty_var.get())
        except ValueError:
            messagebox.showerror("Error", "Price and Quantity must be numbers!")
            return
            
        if not barcode or not product or price <= 0 or qty <= 0:
            messagebox.showerror("Error", "Invalid product information!")
            return
            
        # Cek apakah produk sudah ada di cart
        for i, item in enumerate(self.cart):
            if item["barcode"] == barcode:
                # Update quantity
                self.cart[i]["qty"] += qty
                self.update_cart_table()
                self.update_total()
                return
                
        # Add new item to cart
        self.cart.append({
            "barcode": barcode,
            "product": product,
            "price": price,
            "qty": qty
        })
        
        self.update_cart_table()
        self.update_total()
        
    def update_cart_table(self):
        # Update cart table
        for item in self.cart_table.get_children():
            self.cart_table.delete(item)
            
        for item in self.cart:
            subtotal = item["price"] * item["qty"]
            self.cart_table.insert("", tk.END, values=(
                item["product"],
                f"Rp {item['price']:,.0f}",
                item["qty"],
                f"Rp {subtotal:,.0f}"
            ))
            
    def update_total(self):
        # Update total amount
        total = sum(item["price"] * item["qty"] for item in self.cart)
        self.total_var.set(f"Rp {total:,.0f}")
        
    def remove_item(self):
        # Remove selected item from cart
        selected_item = self.cart_table.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "Please select an item to remove!")
            return
            
        # Get index of selected item
        index = self.cart_table.index(selected_item[0])
        if 0 <= index < len(self.cart):
            del self.cart[index]
            self.update_cart_table()
            self.update_total()
            
    def checkout(self):
        # Process checkout
        if not self.cart:
            messagebox.showwarning("Warning", "Cart is empty!")
            return
            
        # Prepare receipt
        total = sum(item["price"] * item["qty"] for item in self.cart)
        
        receipt = f"===== RECEIPT =====\n"
        receipt += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        receipt += f"{'Product':<20} {'Price':<10} {'Qty':<5} {'Subtotal':<10}\n"
        receipt += "-" * 50 + "\n"
        
        for item in self.cart:
            subtotal = item["price"] * item["qty"]
            receipt += f"{item['product']:<20} {item['price']:<10,.0f} {item['qty']:<5} {subtotal:<10,.0f}\n"
            
        receipt += "-" * 50 + "\n"
        receipt += f"TOTAL: Rp {total:,.0f}\n\n"
        receipt += "Thank you for shopping!\n"
        receipt += "Cashier System by Afra Fadhma Dinata"
        
        # Save receipt to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        receipt_file = f"receipt_{timestamp}.txt"
        
        try:
            with open(receipt_file, "w") as f:
                f.write(receipt)
                
            # Show receipt
            messagebox.showinfo("Checkout Complete", f"Receipt saved to {receipt_file}")
            
            # Clear cart
            self.cart = []
            self.update_cart_table()
            self.update_total()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error saving receipt: {e}")
        
def main():
    # Set tema sesuai OS
    try:
        # Usahain pake tema modern di Windows
        if sys.platform.startswith('win'):
            from tkinter import ttk
            import sv_ttk
            sv_ttk.set_theme("dark")
    except:
        pass
        
    root = tk.Tk()
    app = CashierApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
