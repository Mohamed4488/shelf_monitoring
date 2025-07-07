import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from importnb import Notebook
from utils.data_class import Shelf, Product

with Notebook():
    from notebooks.shelf import  draw_shelves
    from notebooks.product import draw_products

st.set_page_config("Shelf & Product Detection")
st.title("Shelf & Product Detection")

@st.cache_resource
def load_models():
    return YOLO("models/shelf.pt"), YOLO("models/product.pt")

shelf_model, product_model = load_models()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is not None:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        with st.spinner("Detecting shelves and products"):
            shelf_result = shelf_model.predict(img, conf=0.25)[0]
            product_result = product_model.predict(img, conf=0.25)[0]

            shelves = []
            if shelf_result.obb and shelf_result.obb.xyxyxyxy is not None:
                for box in shelf_result.obb.xyxyxyxy:
                    shelves.append(Shelf(box))

            all_products = []
            if product_result.boxes and product_result.boxes.xyxy is not None:
                for box in product_result.boxes.xyxy:
                    all_products.append(Product(box))

            def is_inside(shelf, product):
                px1, py1 = product.p1
                px2, py2 = product.p2

                x_coords = [p[0] for p in [shelf.p1, shelf.p2, shelf.p3, shelf.p4]]
                y_coords = [p[1] for p in [shelf.p1, shelf.p2, shelf.p3, shelf.p4]]
                sx1, sx2 = min(x_coords), max(x_coords)
                sy1, sy2 = min(y_coords), max(y_coords)

                return (sx1 <= px1 <= sx2 and sy1 <= py1 <= sy2) and \
                       (sx1 <= px2 <= sx2 and sy1 <= py2 <= sy2)

            filtered_products = [
                product for product in all_products
                if any(is_inside(shelf, product) for shelf in shelves)
            ]

            combined_img = img.copy()
            if shelves:
                combined_img = draw_shelves(combined_img, shelves)

            if filtered_products:
                combined_img = draw_products(combined_img, filtered_products)
            elif all_products:
                combined_img = draw_products(combined_img, all_products)

            def polygon_area(corners):
                x = [p[0] for p in corners]
                y = [p[1] for p in corners]
                return 0.5 * abs(sum(x[i]*y[(i+1)%4] - x[(i+1)%4]*y[i] for i in range(4)))

            def box_area(p1, p2):
                return abs((p2[0] - p1[0]) * (p2[1] - p1[1]))

            if filtered_products:
                avg_area = np.mean([box_area(p.p1, p.p2) for p in filtered_products])
            else:
                avg_area = 1  

            total_extra_capacity = 0
            
            for shelf in shelves:
                shelf_area = polygon_area([shelf.p1, shelf.p2, shelf.p3, shelf.p4])
                products_in_shelf = [p for p in filtered_products if is_inside(shelf, p)]
                estimated_total = int(shelf_area // avg_area)
                extra = max(0, estimated_total - len(products_in_shelf))
                total_extra_capacity += extra


            st.image(combined_img, use_container_width=True)
            st.write(f"Detected {len(shelves)} shelves")
            st.write(f"Detected  {len(filtered_products)} products")
            st.write(f"Estimated additional products {total_extra_capacity}")
            