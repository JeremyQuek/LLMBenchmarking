# For Functionaliy
import ast
import time
import csv
import pandas as pd
import tqdm._tqdm_pandas

# BERT Imports
from bert_score import score
from datasets import Dataset

# RAGAS PImports
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
# ROUGE Imports
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For KDE
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from scipy.integrate import quad

# for HTML
from jinja2 import Template
import pickle

ragas_metrics = [
    answer_correctness,
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]

other_metrics = [
    "BERT",
    "ROUGE",
    "MRR"

]

class FileContentError(Exception):
    pass

class FileContentKeyError(Exception):
    pass

class FileTypeError(Exception):
    pass


def find_threshold(data):
    kde = stats.gaussian_kde(data)
    x, total_area = 0, 0
    while total_area < 0.1:
        total_area = quad(kde, -0.2, x)[0]
        x +=0.01
    while 0.1 <= total_area < 0.15:
        total_area = quad(kde, -0.2, x)[0]
        x +=0.001
    return x


def create_kde_plot(data, filename, title='SCORE PROBABILITY DISTRIBUTION', xlabel='SCORE',
    ylabel='density', background_color='#272D44', file_format='png', dpi=200, dataset = None):
    # Convert data to numpy array if it's a list
    if isinstance(data, list):
        data = np.array(data)


    # Create the plot with the specified background color
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=background_color)
    ax.set_facecolor(background_color)

    palette = sns.color_palette("crest", 3)

    if data is not None:
        # Create the KDE plot
        sns.kdeplot(data=data, color="#57A0E5", fill=True, ax=ax, label='Density')

    if dataset is not None:
        for i, (key, value) in enumerate(dataset.items()):
            sns.kdeplot(data=value, 
                        fill=True, 
                        ax=ax,
                        color=palette[i], 
                        linewidth=1, 
                        label=key)

    label_colour = '#6A6968'

    # Set labels and title
    ax.set_title(title, color=label_colour, fontsize =15)
    ax.set_xlabel(xlabel, color=label_colour, fontsize =12)
    ax.set_ylabel(ylabel, color=label_colour, fontsize =12)
    x_min, x_max = ax.get_xlim()

    # Set axis limit to 0.0 and 1.0, but expand if data extends beyond
    ax.set_xlim(min(0.0, x_min), max(1.0, x_max))

    # Change tick colors
    ax.tick_params(axis='x', colors=label_colour, labelsize=12)
    ax.tick_params(axis='y', colors=label_colour, labelsize=12)

    # Change spine colors
    for spine in ax.spines.values():
        spine.set_edgecolor('#272D44')

    if data is not None:
        # Find the threshold for 85% area
        threshold = find_threshold(data)
        kde = stats.gaussian_kde(data)
        x = quad(kde, threshold, np.inf)[0]

        # Add a vertical line at the threshold
        ax.axvline(x=threshold, color='red', linestyle='--', label=f"85% Probability Threshold: {round(threshold, 2)}")


    # Ensure the background color extends to the edge of the figure
    plt.tight_layout()

    # Add grid
    ax.grid(color='#59585B', linestyle='--', linewidth=0.5, alpha=0.3)

    # Legend
    ax.legend(facecolor=background_color, edgecolor='none', fontsize=12, labelcolor=label_colour)


    # Save the plot
    full_filename = os.path.join("Dashboard",
                                 f"{filename}.{file_format}")
    plt.savefig(full_filename, format=file_format, dpi=dpi, bbox_inches='tight', facecolor=background_color,
                pad_inches=0.2)
    # Close the plot to free up memory
    plt.close(fig)

    # Return the path to the saved file and the calculated area
    return os.path.abspath(full_filename)


class Dataloader:
    def __new__(cls, dataset=None, stop_word="<END>"):
        instance = super(Dataloader, cls).__new__(cls)
        instance.__init__(dataset, stop_word)
        return instance.results_loaded

    def __init__(self, dataset=None, stop_word="<END>"):
        self.stop_word = stop_word

        if isinstance(dataset, dict):
            self.results_loaded = dataset

        elif isinstance(dataset, pd.DataFrame):
            self.results_loaded = {col: dataset[col].tolist() for col in dataset.columns}

        elif isinstance(dataset, str):
            if dataset.endswith('.xlsx'):
                self.results_loaded = self.load_xl(dataset, self.stop_word)

            elif dataset.endswith('.csv'):
                self.results_loaded = self.load_csv(dataset)

            elif dataset.endswith('.pkl'):
                self.results_loaded = self.load_pkl(dataset)

            else:
                raise FileTypeError

        else:
            raise FileTypeError
        
        self._validate_keys()

    def load_xl(self, dataset, stop_word, sheet_name=0):
        # Read the Excel file
        df = pd.read_excel(dataset, sheet_name=sheet_name)
        
        # Keep only the specified columns
        columns_to_keep = ["answer", "ground_truth", "contexts", "question"]
        df = df[df.columns.intersection(columns_to_keep)]

        # Special handling for 'contexts' column
        if 'contexts' in df.columns:
            df['contexts'] = df['contexts'].apply(lambda x: self.split_contexts(x, stop_word))
        
        results_loaded = {col: df[col].tolist() for col in df.columns}
        return results_loaded
 
    @staticmethod
    def split_contexts(text, stop_word):
        if pd.isna(text):
            return []
        contexts = text.split(stop_word)
        return [context.strip() for context in contexts if context.strip()]
            

    def load_csv(self, dataset):
        results_loaded = {}
        with open(dataset, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)

            for row in reader:
                for key, value in row.items():
                    if key not in results_loaded:
                        results_loaded[key] = []

                    if key == 'contexts':
                        results_loaded[key].append(ast.literal_eval(value))

                    elif key in ['answer', 'ground_truth', 'question']:
                        results_loaded[key].append(value)

        return results_loaded

    def load_pkl(self, dataset):
        with open(dataset, 'rb') as f:
            results_loaded = pickle.load(f)

        if isinstance(results_loaded, dict):
            self.results_loaded = results_loaded

        elif isinstance(results_loaded, pd.DataFrame):
            self.results_loaded = {col: results_loaded[col].tolist() for col in results_loaded.columns}

        else:
            raise FileContentError

        return results_loaded

    def _validate_keys(self):
        required_keys = ['contexts', 'question', 'answer', 'ground_truth']
        if not all(key in self.results_loaded for key in required_keys):
            raise FileContentKeyError
        print("Dataset successfully loaded!")
  

class CalculateMetrics:

    def __new__(cls, results_loaded, llm, embeddings, ragas_metrics, other_metrics):
        instance = super(CalculateMetrics, cls).__new__(cls)
        instance.__init__(results_loaded, llm, embeddings, ragas_metrics, other_metrics)
        instance.generate_report()
        return instance.metric_scores, instance.df, instance.bert_dict
    
    def __init__(self, results_loaded, llm,  embeddings, ragas_metrics, other_metrics):
        self.llm=llm
        self.embeddings=embeddings
        self.results_loaded = results_loaded
        self.ragas_metrics = ragas_metrics
        self.other_metrics = other_metrics

        #Create a dictionary to house all the final values
        self.metric_scores = {}

        #Create a dataframe to house the final dataset
        self.df = pd.DataFrame(self.results_loaded) 

        self.bert_dict = {}

    def ragas_metrics_calc(self, results, metrics):
        dataset = Dataset.from_dict(results)
        #Initialiase
        if self.llm:
            if self.embeddings:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=self.llm,
                    embeddings = self.embeddings,
                    raise_exceptions=False
                )
            else:
                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=self.llm,
                    raise_exceptions=False
                )
        elif self.embeddings:
            result = evaluate(
                    dataset,
                    metrics=metrics,
                    embeddings = self.embeddings,
                    raise_exceptions=False
                )
        #Default LLM
        else:
            result = evaluate(
                dataset,
                metrics=metrics,
                raise_exceptions=False
            )

        return result

    ##DEF Rouge Func
    def rogue_calc(self, contexts, answer):

        scorer = rouge_scorer.RougeScorer(['rougeLsum'], use_stemmer=True)

        highest_f_measure = 0

        for context in contexts:

            scores = scorer.score(str(context), answer)

            f_measure = scores['rougeLsum'].fmeasure

            if f_measure > highest_f_measure:
                highest_f_measure = f_measure

        return highest_f_measure

    # Def BERT Func
    def bert_calc(self, references, candidates):

        if 'Precision' not in self.bert_dict.keys():
            self.bert_dict['Precision'] = []
        
        if 'Recall' not in self.bert_dict.keys():
            self.bert_dict['Recall'] = []

        if 'F1' not in self.bert_dict.keys():
            self.bert_dict['F1'] = []

        P, R, F1 = score([candidates], [references], lang='en', verbose=False)

        self.bert_dict['Precision'].append(P.mean().item())
        self.bert_dict['Recall'].append(R.mean().item())
        self.bert_dict['F1'].append(F1.mean().item())

        return (P.mean().item(), R.mean().item(), F1.mean().item())
    
    # Def MRR Func
    def mrr_calc(self, contexts, ground_truth):
        vectorizer = TfidfVectorizer(stop_words='english')

        # Fit vectorizer on all contexts and the answer
        vectorizer.fit(contexts + [ground_truth])

        # Transform the answer
        answer_vector = vectorizer.transform([ground_truth])

        for i, context in enumerate(contexts):
            # Transform the current context
            context_vector = vectorizer.transform([context])

            # Calculate similarity
            similarity = cosine_similarity(answer_vector, context_vector)[0][0]
            # If similarity threshold is met, return reciprocal rank
            if similarity >= 0.3:
                return float(1 / (i + 1))

        # If no context meets the threshold, return 0
        return 0

    def rate_limit_handler(self):

        #Split the dictionaries into size of 15 maximum to prevent overload
        dicts = []
        for i in range(0, len(next(iter(self.results_loaded.values()))), 15):
            placeholder = {}
            for key, value in self.results_loaded.items():
                placeholder[key] = value[i:i+15]
            dicts.append(placeholder)
        
        
        i = 1
        #Iterate through each metric
        for metric in self.ragas_metrics:
            count = len(self.ragas_metrics)

            #Initilise an empty list to contain all scores
            scores = []
            #For each dictionary per metric
            for dict in dicts:

                #Get the scores
                metric_score = self.ragas_metrics_calc(dict, [metric])

                for metric_name, score in metric_score.items():
                    #If the thing not creted, make it
                    if metric_name not in self.metric_scores:
                        self.metric_scores[metric_name] = []
                    #Append the score in 
                    self.metric_scores[metric_name].append(score)
                
                #Converting the original metric_score gets all the results list
                df = metric_score.to_pandas()
                #Extend the scores to the list
                scores.extend(df[metric_name].to_list())

            self.df[metric_name] = scores

            #Take the average of scores and replace the name
            self.metric_scores[metric_name] = np.mean(self.metric_scores[metric_name])

            i+=1
            #Sleep to save api requests yeah?
            if i <count:
                time.sleep(20)
            

    def generate_report(self):
      
        # Calculate RAGAS scores 1 by 1 to prevent hitting token limit
        print("Calculating RAGAS scores")
        self.rate_limit_handler()

        # Calculate BERT SCORE
        if "BERT" in self.other_metrics:

            tqdm.tqdm.pandas(desc="Calculating BERT scores")

            bert_scores = self.df.progress_apply(lambda row: self.bert_calc(row['ground_truth'], row['answer']), axis=1)
    
            # As bert_calc returns a tuple (precision, recall, f1)
            self.df['BERT Precision'] = bert_scores.apply(lambda x: x[0])
            self.df['BERT Recall'] = bert_scores.apply(lambda x: x[1])
            self.df['BERT F1'] = bert_scores.apply(lambda x: x[2])

            average_BERT_f1 = self.df['BERT F1'].mean()

            self.metric_scores['BERT F1'] = average_BERT_f1

        # Calculate ROUGE SCORE
        if "ROUGE" in self.other_metrics:

            tqdm.tqdm.pandas(desc="Calculating ROUGE scores")

            self.df['Rouge'] = self.df.progress_apply(lambda row: self.rogue_calc(row['contexts'], row['answer']), axis=1)

            average_rougeLsum_fmeasure = self.df['Rouge'].mean()

            self.metric_scores['Rouge'] = average_rougeLsum_fmeasure

        # Calculate MRR SCORE
        if "MRR" in self.other_metrics:

            tqdm.tqdm.pandas(desc="Calculating MRR scores")

            self.df['MRR'] = self.df.progress_apply(lambda row: self.mrr_calc(row['contexts'], row['ground_truth']), axis=1)

            average_MRR = self.df['MRR'].mean()

            self.metric_scores['MRR'] = average_MRR

   
        self.df.to_excel("rag_evaluation_dataset.xlsx", index=False)

        metric_scores = self.metric_scores
        df = self.df
        bert_dict = self.bert_dict

        return metric_scores, df, bert_dict


class EvaluateModel:
    def __init__(self, dataset=None, llm=None, embeddings = None, ragas_metrics=ragas_metrics, other_metrics=other_metrics):
        self.results_loaded = Dataloader(dataset)
        self.ragas_metrics = ragas_metrics
        self.other_metrics = other_metrics

        self.llm=llm if llm else None

        self.embeddings=embeddings if embeddings else None

        self.metric_scores, self.df, self.all_bert_scores = CalculateMetrics(self.results_loaded, 
                                                       self.llm,
                                                       self.embeddings,
                                                       self.ragas_metrics,
                                                       self.other_metrics                  
                                                       )
        
        # Create KDE and FREQ Plots
        def freq_dist(data):
            # Initialize a list of 20 bins (0-0.05, 0.05-0.10, ..., 0.95-1.0)
            bins = [0] * 20

            for value in data:
                if 0 <= value <= 1:
                    # Calculate the bin index
                    bin_index = min(int(value * 20), 19)
                    bins[bin_index] += 1

            return bins

        if 'answer_correctness' in self.metric_scores.keys():
            ac_dist = freq_dist(self.df['answer_correctness'].dropna().tolist())
            create_kde_plot(self.df['answer_correctness'].dropna().tolist(), filename='my_kde_plot1')
        else:
            ac_dist = [0]*20

        if 'faithfulness' in self.metric_scores.keys():
            ff_dist = freq_dist(self.df['faithfulness'].dropna().tolist())
            create_kde_plot(self.df['faithfulness'].dropna().tolist(), filename='my_kde_plot2')
        else:
            ff_dist = [0]*20

        if 'answer_relevancy' in self.metric_scores.keys():
            ar_dist = freq_dist(self.df['answer_relevancy'].dropna().tolist())
            create_kde_plot(self.df['answer_relevancy'].dropna().tolist(), filename='my_kde_plot3')
        else:
            ar_dist= [0]*20

        if 'context_recall' in self.metric_scores.keys():
            cr_dist = freq_dist(self.df['context_recall'].dropna().tolist())
            create_kde_plot(self.df['context_recall'].dropna().tolist(), filename='my_kde_plot4')
        else:
            cr_dist = [0]*20

        if 'context_precision' in self.metric_scores.keys():
            cp_dist = freq_dist(self.df['context_precision'].dropna().tolist())
            create_kde_plot(self.df['context_precision'].dropna().tolist(), filename='my_kde_plot5')
        else:
            cp_dist = [0]*20


        if 'BERT F1' in self.metric_scores.keys():
            bs_dist = freq_dist(self.df['BERT F1'].dropna().tolist())
            create_kde_plot(dataset = self.all_bert_scores, filename='my_kde_plot6', data = None)
        else:
            bs_dist = [0]*20

        if 'Rouge' in self.metric_scores.keys():
            rg_dist = freq_dist(self.df['Rouge'].dropna().tolist())
            create_kde_plot(self.df['Rouge'].dropna().tolist(), filename='my_kde_plot7')
        else:
            rg_dist = [0]*20

        if 'MRR' in self.metric_scores.keys():
            mrr_dist = freq_dist(self.df['MRR'].dropna().tolist())
            create_kde_plot(self.df['MRR'].dropna().tolist(), filename='my_kde_plot8')
        else:
            mrr_dist = [0]*20

        # Create HTML Dashboard
        @staticmethod
        def dashboard():
            if 'answer_correctness' in self.metric_scores.keys():
                ac = round(self.metric_scores['answer_correctness'], 3) * 100
            else:
                ac = None
            if 'faithfulness' in self.metric_scores.keys():
                ff = round(self.metric_scores['faithfulness'], 3) * 100
            else:
                ff = None
            if 'answer_relevancy' in self.metric_scores.keys():
                ar = round(self.metric_scores['answer_relevancy'], 3) * 100
            else:
                ar = None
            if 'context_precision' in self.metric_scores.keys():
                cp = round(self.metric_scores['context_precision'], 3) * 100
            else:
                cp = None
            if 'context_recall' in self.metric_scores.keys():
                cr = round(self.metric_scores['context_recall'], 3) * 100
            else:
                cr = None
            if 'BERT F1' in self.metric_scores.keys():
                bs = round(self.metric_scores['BERT F1'], 3) *100
            else:
                bs = None
            if 'Rouge' in self.metric_scores.keys():
                rg = round(self.metric_scores['Rouge'], 3) *100
            else:
                rg = None
            if 'MRR' in self.metric_scores.keys():
                mrr = round(self.metric_scores['MRR'], 3) * 100
            else:
                mrr = None

            #Add a S/N column
            self.df['S/N'] = range(1, len(self.df) + 1)
            cols = ['S/N'] + [col for col in self.df if col != 'S/N']
            df = self.df[cols]

            df_html = df.to_html(classes='dataframe', index=False)

            # Define the HTML template
            from HTML_template import html_template
            html_template = html_template

            # Create a Template object
            template = Template(html_template)

            # Render the template with any necessary variables
            html_content = template.render(ac=ac,
                                           ff=ff,
                                           ar=ar,
                                           cr=cr,
                                           cp=cp,
                                           rg=rg,
                                           bs=bs,
                                           mrr=mrr,

                                           ac_dist=ac_dist,
                                           ff_dist=ff_dist,
                                           ar_dist=ar_dist,
                                           cr_dist=cr_dist,
                                           cp_dist=cp_dist,
                                           rg_dist=rg_dist,
                                           bs_dist=bs_dist,
                                           mrr_dist=mrr_dist,

                                           df_html=df_html,
                                           df=df,
                                           )

            # Write the HTML content to a file
            with open(
                    'Dashboard/evaluation_report_card.html',
                    'w') as file:
                file.write(html_content)

            print("HTML file generated successfully!")

        #Call the dashboard functions above
        dashboard()

        return 

    # Print and get functions
    def __str__(self):
        if self.metric_scores is None:
            return "Metrics have not been calculated yet."
        return self.format_metrics()

    def format_metrics(self):
        if self.metric_scores is None:
            return "Metrics have not been calculated yet."

        metrics_str = "Metrics:\n"
        for key, value in self.metric_scores.items():
            metrics_str += f"{key}: {value}\n"
        return metrics_str

    def get_dataset(self):
        return self.df


# Stuff to consider: 1) hwo to pass a critic model in
# how to load a ROBERTA-LARGE model
# How to receieve and accept files
# how to pass html files