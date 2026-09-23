import os

def generate_maxi_files():
    maxi_authors_path = "Maxi_AUTHORS.md"
    maxi_python_path = "Maxi_Python.py"
    maxi_readme_path = "Maxi_README.md"

    # Ouverture des fichiers de sortie en écriture
    with open(maxi_authors_path, "w", encoding="utf-8") as f_authors, \
         open(maxi_python_path, "w", encoding="utf-8") as f_python, \
         open(maxi_readme_path, "w", encoding="utf-8") as f_readme:

        # Parcours des éléments de niveau 1 dans le dossier courant
        for entry in sorted(os.listdir(".")):
            # On ignore le dossier .git et les fichiers/dossiers cachés
            if entry.startswith("."):
                continue
            
            # On s'assure de ne traiter que les sous-dossiers (niveau 1)
            if os.path.isdir(entry):
                group_name = entry
                
                # Recherche du fichier AUTHORS (gestion de AUTHORS ou AUTHORS.md)
                authors_file = None
                for filename in ["AUTHORS", "AUTHORS.md"]:
                    candidate = os.path.join(group_name, filename)
                    if os.path.exists(candidate):
                        authors_file = candidate
                        break

                # Recherche du fichier README (gestion de la casse et extension)
                readme_file = None
                for filename in ["README.md", "README", "readme.md", "Readme.md"]:
                    candidate = os.path.join(group_name, filename)
                    if os.path.exists(candidate):
                        readme_file = candidate
                        break
                
                # --- 1. Concaténation pour Maxi_AUTHORS.md ---
                f_authors.write(f"## {group_name}\n\n")
                if authors_file:
                    with open(authors_file, "r", encoding="utf-8") as af:
                        f_authors.write(af.read().strip() + "\n")
                else:
                    f_authors.write("*Fichier AUTHORS absent*\n")
                f_authors.write("\n---\n\n")

                # --- 2. Concaténation pour Maxi_Python.py ---
                f_python.write(f"# NOM DU GROUPE : {group_name}\n")
                
                if authors_file:
                    with open(authors_file, "r", encoding="utf-8") as af:
                        f_python.write(af.read().strip() + "\n")
                
                snake_file = os.path.join(group_name, "snake.ia.py")
                if not os.path.exists(snake_file):
                    alt_snake = os.path.join(group_name, "snake-ia.py")
                    if os.path.exists(alt_snake):
                        snake_file = alt_snake

                if os.path.exists(snake_file):
                    with open(snake_file, "r", encoding="utf-8") as sf:
                        f_python.write(sf.read().strip() + "\n")
                else:
                    f_python.write("# (Fichier snake.ia.py non trouvé pour ce groupe)\n")
                
                f_python.write("\n\n" + "# " + "="*40 + "\n\n")

                # --- 3. Concaténation pour Maxi_README.md ---
                f_readme.write(f"# NOM DU GROUPE : {group_name}\n\n")
                
                if authors_file:
                    with open(authors_file, "r", encoding="utf-8") as af:
                        f_readme.write(af.read().strip() + "\n\n")
                
                if readme_file:
                    with open(readme_file, "r", encoding="utf-8") as rf:
                        f_readme.write(rf.read().strip() + "\n")
                else:
                    f_readme.write("*Fichier README absent*\n")
                
                f_readme.write("\n\n" + "-"*40 + "\n\n")

    print("Génération des fichiers Maxi_AUTHORS.md, Maxi_Python.py et Maxi_README.md terminée avec succès !")

if __name__ == "__main__":
    generate_maxi_files()