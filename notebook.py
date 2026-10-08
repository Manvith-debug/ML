"""Serif experiment notebook, drawn directly as vector PDF with FPDF2."""
import json
import math
import numpy as np
import pandas as pd
from fpdf import FPDF

INK = (31, 51, 42)
GREEN = (41, 110, 83)
OCHRE = (172, 112, 39)
GREY = (105, 110, 105)


class Notebook(FPDF):
    def header(self):
        self.set_draw_color(*GREEN)
        self.set_line_width(.8)
        self.line(18, 15, 192, 15)
        self.set_xy(18, 18)
        self.set_font('Courier', '', 9)
        self.set_text_color(*GREEN)
        self.cell(174, 5, 'GEOTHERMAL MODELS / BT2024261')
        self.set_xy(18, 30)

    def footer(self):
        self.set_y(-16)
        self.set_font('Times', 'I', 9)
        self.set_text_color(*GREY)
        self.cell(174, 5, f'Experiment notebook  |  8 October 2026  |  {self.page_no()}/{{nb}}', align='C')

    def chapter(self, number, title, subtitle):
        self.add_page()
        self.set_text_color(*INK)
        self.set_font('Times', '', 27)
        self.multi_cell(174, 12, f'{number} / {title}', new_x='LMARGIN', new_y='NEXT')
        self.set_font('Times', 'I', 12)
        self.multi_cell(174, 6, subtitle, new_x='LMARGIN', new_y='NEXT')
        self.ln(6)

    def heading(self, text):
        self.set_text_color(*GREEN)
        self.set_font('Helvetica', 'B', 11)
        self.multi_cell(174, 6, text, new_x='LMARGIN', new_y='NEXT')
        self.ln(1)

    def paragraph(self, text, small=False):
        self.set_text_color(*INK)
        self.set_font('Times', '', 10 if small else 12)
        self.multi_cell(174, 5 if small else 6, text, new_x='LMARGIN', new_y='NEXT')
        self.ln(3)

    def grid(self, rows, widths):
        for i, row in enumerate(rows):
            self.set_font('Helvetica', 'B' if i == 0 else '', 9)
            self.set_fill_color(*(GREEN if i == 0 else (243, 246, 241)))
            self.set_text_color(*( (255,255,255) if i == 0 else INK))
            for value, width in zip(row, widths):
                self.cell(width, 9, str(value), fill=True)
            self.ln(9)
        self.ln(5)

    def code(self, lines):
        self.set_font('Courier', '', 9)
        self.set_text_color(*INK)
        self.set_fill_color(243,246,241)
        self.multi_cell(174, 5, '\n'.join(lines), fill=True, new_x='LMARGIN', new_y='NEXT')
        self.ln(5)


def degree_chart(pdf, x, y, width, height, search, maximum, title):
    left, top, w, h = x+10, y+8, width-13, height-20
    dense = search[search.method == 'ridge']
    lower = math.floor(math.log10(dense.mean_mse.min()))
    upper = math.ceil(math.log10(dense.mean_mse.max()))
    def px(degree): return left + (degree-1)/(maximum-1)*w
    def py(error): return top + h - (math.log10(error)-lower)/(upper-lower)*h
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*INK)
    pdf.set_xy(x, y)
    pdf.cell(width, 5, title)
    for exponent in range(lower, upper+1):
        level = py(10**exponent)
        pdf.set_draw_color(210,218,212)
        pdf.set_line_width(.2)
        pdf.line(left, level, left+w, level)
        pdf.set_xy(x, level-2)
        pdf.set_font('Helvetica', '', 7)
        pdf.cell(9, 4, f'{10**exponent:g}', align='R')
    for basis, color in [('power',GREEN),('chebyshev',OCHRE)]:
        d = dense[dense.basis == basis].sort_values('degree')
        points = [(px(a),py(b)) for a,b in zip(d.degree,d.mean_mse)]
        pdf.set_draw_color(*color)
        pdf.set_fill_color(*color)
        pdf.set_line_width(.6)
        for a,b in zip(points,points[1:]):
            pdf.line(a[0],a[1],b[0],b[1])
        for xx,yy in points:
            pdf.ellipse(xx-.45,yy-.45,.9,.9,style='F')
    pdf.set_text_color(*INK)
    pdf.set_font('Helvetica','',8)
    for degree in (1,maximum//2,maximum):
        pdf.set_xy(px(degree)-3,top+h+2)
        pdf.cell(6,4,str(degree),align='C')
    pdf.set_xy(left,top+h+7)
    pdf.cell(w,4,'Polynomial degree',align='C')


def scatter(pdf, x, y, width, height, data, title):
    left, top, w, h = x+12,y+8,width-15,height-23
    lo = math.floor(min(data.observed.min(),data.estimated.min())/5)*5
    hi = math.ceil(max(data.observed.max(),data.estimated.max())/5)*5
    def px(v): return left+(v-lo)/(hi-lo)*w
    def py(v): return top+h-(v-lo)/(hi-lo)*h
    pdf.set_text_color(*INK)
    pdf.set_font('Helvetica','B',10)
    pdf.set_xy(x,y)
    pdf.cell(width,5,title)
    pdf.set_line_width(.2)
    for value in np.linspace(lo,hi,3):
        pdf.set_draw_color(215,221,216)
        pdf.line(left,py(value),left+w,py(value))
        pdf.line(px(value),top,px(value),top+h)
        pdf.set_font('Helvetica','',7)
        pdf.set_xy(x,py(value)-2)
        pdf.cell(10,4,f'{value:g}',align='R')
        pdf.set_xy(px(value)-4,top+h+2)
        pdf.cell(8,4,f'{value:g}',align='C')
    pdf.set_draw_color(*OCHRE)
    pdf.set_line_width(.5)
    pdf.line(px(lo),py(lo),px(hi),py(hi))
    pdf.set_fill_color(*GREEN)
    for actual,estimate in zip(data.observed,data.estimated):
        pdf.ellipse(px(actual)-.55,py(estimate)-.55,1.1,1.1,style='F')
    pdf.set_xy(left,top+h+7)
    pdf.set_font('Helvetica','',8)
    pdf.cell(w,4,'Actual score (x); predicted score (y)',align='C')


def build(project, repository='https://github.com/Manvith-debug/ML'):
    result = json.loads((project / 'results.json').read_text())
    problems = result['problems']
    searches = {v: pd.read_csv(project / f'{v}_search.csv') for v in problems}
    pdf = Notebook()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 23)
    pdf.alias_nb_pages()
    pdf.set_title('BT2024261 - Geothermal prediction experiment notebook')
    pdf.set_author('BT2024261')
    pdf.chapter('01', 'The question', 'Which polynomial model predicts each score reliably?')
    pdf.paragraph('This assignment has two separate prediction tasks. The first uses six power-plant settings to predict Net Power Score. The second uses three location coordinates to predict Thermal Anomaly Score. Only the files assigned to BT2024261 were used.')
    pdf.grid([['Dataset','Inputs','Training rows','Test rows'],['var1','x1 to x6',1000,1000],['var2','x1 to x3',1000,1000]], [36,48,45,45])
    pdf.heading('First look at the data')
    pdf.paragraph('There are no missing values, and every input lies between -1 and 1. Var1 has no repeated input vectors; var2 has 11 extra rows with repeated inputs. These observations were retained, and matching inputs were always placed in the same data split.')
    pdf.heading('The modeling idea')
    pdf.paragraph('A polynomial combines powers and interactions of the inputs. For example, x1 squared times x2 has degree 3. The search compared ordinary power terms with Chebyshev terms, a different way to represent the same polynomial space. Both stay within the degree limits of the assignment.')
    pdf.heading('What the experiment selected')
    pdf.grid([['Dataset','Method','Degree','Active terms'], *[[v, 'Sparse OMP' if m['selected']['method']=='omp' else 'Ridge',m['selected']['degree'],m['nonzero_terms']] for v,m in problems.items()]], [36,58,35,45])
    pdf.paragraph(f"Var1 works best with a small set of selected terms. Var2 benefits from a larger polynomial with coefficients kept under control by Ridge regularization; its final penalty alpha is {problems['var2']['alpha']:g}. Ordinary power terms were better than Chebyshev terms on both datasets.")

    pdf.chapter('02', 'The experiment', 'The same five validation folds were used for every candidate.')
    pdf.paragraph('Seed 261 was used to reserve about 20% of input groups for a final holdout check. The remaining rows were divided into five folds. Each candidate was fitted on four folds and checked on the fifth; the average error determined the winner.')
    pdf.paragraph('Stage A tested Ridge at every degree from 1 to 10 for var1 and 1 to 20 for var2, using both types of terms. Ridge chose its penalty from ten values, 0.000001 to 1000, using an internal leave-one-out score. Scaling was fitted only on the current training partition.')
    pdf.paragraph('Stage B tested a sparse method, Orthogonal Matching Pursuit (OMP), at the best Ridge degree for each type of term. OMP adds useful terms one at a time and refits their coefficients. The tested limits were 12, 24, 48 and 96 terms. This focused search did not test OMP at every degree.')
    y = pdf.get_y()
    degree_chart(pdf,18,y,83,64,searches['var1'],10,'var1 / Ridge degree scan')
    degree_chart(pdf,109,y,83,64,searches['var2'],20,'var2 / Ridge degree scan')
    pdf.set_xy(18,y+66)
    pdf.paragraph('Green: ordinary powers. Ochre: Chebyshev terms. The vertical axis shows mean validation MSE on a logarithmic scale; lower is better.',small=True)
    rows = [['Candidate family','var1 best MSE','var2 best MSE']]
    for method,basis,label in [('ridge','power','Ridge / ordinary'),('ridge','chebyshev','Ridge / Chebyshev'),('omp','power','OMP / ordinary'),('omp','chebyshev','OMP / Chebyshev')]:
        rows.append([label,*[f"{d[(d.method==method)&(d.basis==basis)].mean_mse.min():.6f}" for d in searches.values()]])
    pdf.grid(rows,[80,47,47])
    pdf.paragraph('The selected settings had the lowest mean five-fold MSE. Holdout results were not used to change the model choice.',small=True)

    pdf.chapter('03', 'The check', 'Performance on rows kept out of the model search.')
    pdf.paragraph('MSE is the average squared prediction error, so lower values are better. R2 measures how much variation in the score the model explains; values nearer 1 are better. A mean-only baseline predicts the average training score for every row.')
    pdf.grid([['Dataset','Holdout rows','MSE','R2'], *[[v,m['holdout_rows'],f"{m['holdout']['mse']:.6f}",f"{m['holdout']['r2']:.6f}"] for v,m in problems.items()]], [36,48,45,45])
    y = pdf.get_y()
    for x,(var,m) in zip([18,109],problems.items()):
        scatter(pdf,x,y,83,77,pd.read_csv(project / f'{var}_holdout.csv'),var + ' / actual versus predicted')
    pdf.set_xy(18,y+81)
    pdf.paragraph('The line shows perfect predictions. Points near it have small errors. Both models improve on the mean-only baseline.',small=True)
    pdf.heading('What may change on new rows')
    m = problems['var1']
    pdf.paragraph(f"Var1 has more test rows at the input boundaries: {m['test_boundary_fraction']:.1%}, compared with {m['train_boundary_fraction']:.1%} in training. Here, a boundary row has at least one input at -1 or 1. This difference can affect accuracy.")
    pdf.paragraph('The holdout is one split, so its scores are estimates rather than guarantees. The lowest validation score can also be optimistic because many candidates were compared. All 1,000 labeled rows were used for the final model after the holdout check.')

    pdf.chapter('04', 'The hand-off', 'One directory, reproducible settings, and two prediction files.')
    pdf.heading('Software and organization')
    pdf.paragraph('The project uses scikit-learn pipelines for feature creation, scaling and regression. Settings are stored in a TOML file. Each fitted pipeline is saved as a Joblib model. This PDF is produced with FPDF2; its charts are drawn directly as vector graphics.')
    pdf.code(['geopoly.py          command-line interface','features.py         polynomial representations','experiment.py       search, fitting and prediction','checks.py           verification','notebook.py         report generator','settings.toml       experiment settings','results.json        results and data fingerprints','var1.joblib / var2.joblib    saved pipelines','BT2024261_pred_*.csv         submissions'])
    pdf.heading('Optional: reproduce from the project directory')
    pdf.code(['python -m pip install -r requirements.txt','python geopoly.py train','python geopoly.py check','python geopoly.py report'])
    pdf.paragraph('The repository directory must contain the four BT2024261 dataset files and sample_submission.csv. A different folder can be supplied with --data. The saved pipeline also supports prediction without retraining; the README gives the command.',small=True)
    pdf.heading('Files to submit')
    pdf.paragraph('BT2024261_pred_var1.csv and BT2024261_pred_var2.csv each contain 1,000 rows and one column named y. The original test order is preserved. No prediction clipping is applied. The check command verifies the splits, fitted models, scores and CSV structure.',small=True)
    pdf.heading('Repository')
    pdf.paragraph(repository if repository else 'Repository link to be added.',small=True)
    pdf.paragraph('Source data: the four supplied BT2024261 CSV files. Dataset fingerprints and software versions are recorded in results.json.',small=True)
    if pdf.page_no() != 4:
        raise RuntimeError(f'Expected four pages, got {pdf.page_no()}')
    path = project / 'BT2024261_experiment_notes.pdf'
    pdf.output(path)
    print(f'Created {path}')


