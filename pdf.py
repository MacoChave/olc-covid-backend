from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font("Arial", "I", 8)
        # Title
        self.cell(0, 9, self.title, 0, 1, "R")
        # Author
        self.cell(0, 9, self.author, 0, 1, "R")
        # Line break
        self.ln(10)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("Arial", "I", 8)
        # Text color in gray
        # self.set_text_color(128)
        # Page number
        self.cell(0, 10, "Page " + str(self.page_no()), 0, 0, "C")

    def chapter_title(self, num, label):
        # Arial 12
        self.set_font("Arial", "B", 14)
        # Background color
        # self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, "%d. %s" % (num, label), 0, 1, "C")
        # Line break
        self.ln(4)

    def chapter_body(self, name, type):
        if type == "text":
            # Read text file
            with open(name, "rb") as fh:
                txt = fh.read().decode("latin-1")
            # Times 12
            self.set_font("Arial", "", 12)
            # Output justified text
            self.multi_cell(0, 5, txt)
        else:
            self.image(name, 30, 70, 150)
        # Line break
        self.ln()
        # Mention in italics
        # self.set_font("", "I")
        # self.cell(0, 5, "(end of excerpt)")

    def print_chapter(self, num, title, name, type="text"):
        self.add_page()
        self.chapter_title(num, title)
        self.chapter_body(name, type)


# ESCRIBIR des.txt, datos.txt, conc.txt
def genPDF(title, image):
    pdf = PDF()
    pdf.set_title(title)
    pdf.set_author("Marco Antonio Fidencio Chávez Fuentes")
    pdf.print_chapter(1, "Nomenclatura", "./reporte/nom.txt")
    pdf.print_chapter(2, "Introducción", "./reporte/intro.txt")
    pdf.print_chapter(3, "Desarrollo", "./reporte/des.txt")
    pdf.print_chapter(4, "Datos calculados", "./reporte/datos.txt")
    pdf.print_chapter(5, "Concluciones", "./reporte/conc.txt")
    pdf.print_chapter(6, "Anexos", image, "img")
    pdf.output("./reporte/201020831_reporte.pdf", "F")
