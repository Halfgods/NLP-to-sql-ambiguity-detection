Clarion project report submission

Final submission copy: Project_Report.pdf
Editable report source: Project_Report.html

Note: Project_Report.html and Project_Report.pdf are intentionally kept out of
Git because they contain the team's names and roll numbers. The rest of this
directory (assets, figures, drafts) is safe to publish.

Report order: abstract and problem statement; supplied architecture image and
current request path; measured model and metric comparisons; frontend capture;
live model output with its question, SQL, and returned rows.

Included evidence and assets:
- architecture.png — supplied architecture diagram
- screenshots/frontend.png — current frontend screenshot
- screenshots/model-output.png — live read-only model output and rows
- figures/model_comparison.png — comparison chart used in the report
- model_comparison_results.json — measured comparison values and provenance
- compare_models.py — comparison runner
- generate_figures.py — chart generation script
- References page — papers and official documentation for the cited model and software components

Rebuild the PDF from the editable source (with WeasyPrint installed):
  python3 -c 'from weasyprint import HTML; HTML(filename="Project_Report.html").write_pdf("Project_Report.pdf")'

Regenerate figures from the saved results:
  python3 generate_figures.py

The report records the local checkpoint and held-out dataset scope. MiniLM is
compared on ambiguity detection, where it is used; it is not treated as a
Text-to-SQL generator. The browser screenshots came from a live run on
6 November 2026 using a temporary SQLite workspace under /tmp.

Superseded drafts are archived in drafts/ and are not submission copies.
