import tkinter as tk
from tkinter import filedialog, ttk
import csv


def vcsv():
    root = tk.Tk()
    root.title("Attendance Report")
    root.geometry("1000x600")

    # ===== FILE PICKER =====
    filename = filedialog.askopenfilename(
        title="Select CSV File",
        filetypes=(("CSV Files", "*.csv"),)
    )

    if not filename:
        root.destroy()
        return

    # ===== READ CSV =====
    with open(filename, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)

    # ===== TITLE =====
    tk.Label(root, text="Attendance Report", font=("Helvetica", 20, "bold")).pack(pady=10)

    # ===== TABLE =====
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    tree = ttk.Treeview(frame, columns=header, show="headings")

    # headings
    for col in header:
        tree.heading(col, text=col)
        tree.column(col, width=120, anchor="center")

    # data insert
    for row in data:
        tree.insert("", "end", values=row)

    # scrollbars
    scrollbar_y = tk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar_x = tk.Scrollbar(frame, orient="horizontal", command=tree.xview)

    tree.configure(yscroll=scrollbar_y.set, xscroll=scrollbar_x.set)

    scrollbar_y.pack(side="right", fill="y")
    scrollbar_x.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    # ===== BACK BUTTON =====
    tk.Button(root, text="Back", width=15, command=root.destroy).pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    vcsv()

# end