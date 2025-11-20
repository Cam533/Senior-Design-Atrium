import PyPDF2

def parse_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # turn pdf into txt file
        new_path = pdf_path.replace('.pdf', '.txt')
        with open(new_path, 'w') as txt_file:
            for page in reader.pages:
                txt_file.write(page.extract_text())
        return new_path

def main():
    pdf_path = "C:\Users\Owner\OneDrive\Documents\2025 Fall\CIS 4000\Senior-Design-Atrium\data\Development_Checklist-July-2024.pdf""
    new_path = parse_pdf(pdf_path)
    print(f"PDF parsed and saved to {new_path}")

if __name__ == "__main__":
    main()