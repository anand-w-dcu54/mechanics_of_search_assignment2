import os
import nltk
nltk.download('stopwords')
import pandas as pd
import search_engine as se
from PIL import Image, ImageTk

import streamlit as st
from PIL import Image
from pathlib import Path

import re



#ctk.set_appearance_mode("syste")
#ctk.set_default_color_theme("blue")

def show_images(results):

    # Clear existing images in the frame (if any)
    for widget in scrollable_frame.winfo_children():
        widget.destroy()

    col=0
    row=0

    for iter, image_path in enumerate(get_top_n_file_paths(results,8)):
        photo_images.clear()
        image = Image.open(image_path)
        image = image.resize((400, 400))
        photo = ImageTk.PhotoImage(image)
        photo_images.append(photo)  


        #image_label = ctk.CTkLabel(scrollable_frame, text="", image=photo, compound="top")
        #image_label.grid(row=row, column=col, padx=20, pady=20)
        #image_label.pack(pady=30)
        col += 1
        if col > 2:
            col = 0
            row += 1


def search_query(query):
    
    try:
        
        if not query:
            raise ValueError("Query cannot be empty.")
        
        results = search_engine.search_for_query(query)
        

        if not results:
            raise ValueError("No results found.")
        else:
            print(f"results = {results}")
            #show_images(results)
    except:
        print(f"Error: {query_var.get()}")
        return

    return(results)

def get_top_n_file_paths(results, top_n_items=5, local_run =False):
    """
    Get the top N file paths from the results.
    """
    for item in [k[0] for k in results.items()][:top_n_items]:
        doc_id = item
        file_path = df.loc[df['doc_id'] == item, 'image_path'].values[0]
        if not local_run:
            file_path = re.sub(r'c:\\Users\\User\\Documents\\DCU_MSC\\Semester4\\Mechanics of search\\Assignment_2\\Image_db\\', 'Image_db/', file_path)
            #print(file_path)
        image_caption = df.loc[df['doc_id'] == item, 'caption'].values[0]
        yield [file_path,image_caption]

#App Frame

df = pd.read_csv("image_database.csv", index_col=False)
df = df.fillna('')
df.reset_index(drop=True, inplace=True)
df['doc_id'] = df.index


df['processed_text'] = df['alt_text'].apply(se.preprocess_text) + df['caption'].apply(se.preprocess_text) + df['tags'].apply(se.preprocess_text)
df['processed_text'] = df['processed_text'].apply(se.remove_duplicate_words)

m_index = se.InvertedIndex(df)
m_index.create_index()
search_engine = se.search_engine_lm(m_index)



st.title("Image Search Engine")

# User input
query = st.text_input("Enter your search query:")


num_columns = 3
cols = st.columns(num_columns)

if st.button("Search"):
    if not query:
        st.warning("Please enter a search query.")
    else:
        found_images = search_query(query)
        if found_images:
            #st.success(f"Found {len(found_images)} image(s).")
            for index, img_path in enumerate(get_top_n_file_paths(found_images, 10)):
                if os.path.exists(img_path[0]):
                    image = Image.open(img_path[0])
                    with cols[index % num_columns]:
                        st.image(image, caption=img_path[1], use_container_width=True)
                   
                else:
                    st.error(f"Image path not found: {img_path[0]} This is due to the the limit github sets of allowing only 1000 files to be uploaded to the repo")
                
                #st.image(image, caption=img_path.name, use_column_width=True)
        else:
            st.error("No images found matching your query.")