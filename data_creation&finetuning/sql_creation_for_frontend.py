# run it in colab
# 🗂️ Step 1: Import libraries
import pandas as pd
from google.colab import files

# 🧩 Step 2: Upload your CSV file
print("📤 Please upload your CSV file (e.g., Filtered_Unique_Menu_Items.csv)")
uploaded = files.upload()

# Automatically get the uploaded file name
file_name = list(uploaded.keys())[0]
print(f"✅ Uploaded file: {file_name}")

# 🧾 Step 3: Load the CSV into a DataFrame
df = pd.read_csv(file_name)
print("📊 Preview of your data:")
display(df.head())

# 🧮 Step 4: Generate SQL INSERT statements
output_lines = []

for _, row in df.iterrows():
    category = str(row['Category']).replace("'", "''")
    item_name = str(row['Item_Name']).replace("'", "''")
    sql_line = (
        f"({row['Item_id']}, '{category}', '{item_name}', {row['Price']}, "
        f"'default_image_path.jpg', NULL, NULL)"
    )
    output_lines.append(sql_line)

# 🧱 Step 5: Combine everything into one SQL command
sql_output = (
    "INSERT INTO `menu_items` (`id`, `category`, `dish`, `price`, `image_path`, `created_by`, `modified_by`)\n"
    "VALUES\n" + ",\n".join(output_lines) + ";"
)

# 💾 Step 6: Save as .txt file
output_file = "menu_items_insert_statements.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(sql_output)

print("✅ SQL file created successfully!")

# 📥 Step 7: Download the file
files.download(output_file)
