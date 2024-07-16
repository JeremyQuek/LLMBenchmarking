html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>RAG Evaluation Toolkit</title>
                <link rel ="style sheet" href = "style.css">
                <title>charts</title>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                <style>
                    body {
                        background: #1E1F24;
                        text-align: center;
                        font-family: "Arial", sans serif;
                        color: #CACCDA;
                    }
                    .normal-line-height {
                        line-height: 1.8; 
                    }        



                    .main {
                        max-width: 1300px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1 {
                        font-size: 2.5rem;
                        margin-bottom: 85px;
                    }
                    h4 {
                        font-size: 1.5rem;
                        text-align: left;
                        margin-bottom: 30px;
                    }



                    .flex-row {
                        display: flex;
                        flex-direction: row;
                        justify-content: space-between;
                        margin-bottom: 15px;
                    }
                    .metric-box {
                        background-color: #282631;
                        border-radius: 8px;
                        text-align: center;
                        padding: 25px;
                        width: 15%;
                        margin-bottom: 15px;
                    }
                    .metric-title {
                        font-size: 0.8rem;
                        margin-bottom: 15px;
                    }
                    .metric-value {
                        font-size: 2.2rem;
                        font-weight: bold;
                        margin-bottom: 15px;
                    }
                    .metric-none {
                        font-size: 0.7rem;
                        line-height: 1.8;
                        color: #808080;
                        margin-top: 40px;
                    }
                    progress {
                        width: 100%;
                        height: 10px;
                    }

                    .flex-row2 {
                        display: flex;
                        flex-direction: row;
                        justify-content: space-between;
                        margin-bottom: 15px;
                    }
                    .metric-box2 {
                        background-color: DarkCyan;
                        border-radius: 8px;
                        text-align: center;
                        padding: 25px;
                        width: 28.7%;
                        margin-bottom: 15px;
                    }
                    .metric-none2 {
                        font-size: 0.7rem;
                        line-height: 1.8;
                        margin-top: 40px;
                    }
                    .progress {
                        width: 100%;
                        height: 10px;
                    }



                    .tab {
                    background-color: #24222D;
                    border-radius: 8px;
                    padding: 30px;
                    overflow: hidden;
                    }
                    /* Style the buttons that are used to open the tab content */
                    .tab button {
                      background-color: inherit;
                      float: left;
                      border: none;
                      outline: none;
                      cursor: pointer;
                      padding: 14px 16px;
                      transition: 0.3s;
                      color: #CACCDA;
                      font-size: 0.9rem;
                    }
                    /* Change background color of buttons on hover */
                    .tab button:hover {
                      background-color:#413D50;
                      padding: 30px;
                      border-radius: 12px;
                      border-top: 2px solid orange;
                    }
                    /* Create an active/current tablink class */
                    .tab button.active {
                      background-color:  #413D50;
                      border-radius: 8px;
                    }
                    /* Style the tab content */
                    .tabcontent {
                      display: none;
                      padding: 6px 12px;
                      border-radius: 8px;
                      justify-content: center;
                      border-top: none;
                      background-color: #DFE4EA;
                    }
                    .tabtext {
                        text-align: left;
                        margin-left: 40px;
                        line-height: 1.8;
                        color: #3F4456;
                    }
                    .tabtext2 {
                        text-align: left;
                        margin-left: 40px;
                        line-height: 1.8;
                        color: #63697E;
                        font-size: 1.5rem;
                    }
                    .tabtext3 {
                        text-align: middle;
                        margin-left: 20px;
                        line-height: 1.8;
                        color:  #3F4456;
                    }
                    .recommendation {
                        background-color:  #BCC0C4;
                        color: #3F4456;
                        border-radius: 15px;
                        width: 90%;
                        padding: 30px;
                        text-align: left;
                        line-height:1.5;
                        margin: 0 0 0 2.5%;
                    }
                    .recommendation_text{
                        font-size: 0.7rem;
                    }
                    .chart-container {
                        width: 10%;
                        background-color: inherit;
                        border-radius: 10px;
                        padding: 20px;
                        margin: 0 0 0 7.5%;
                    }




                    .table-container {
                        width: 1350px; /* Adjust as needed */
                        overflow-x: auto;
                        max-width: 8000px;
                        padding: 10px 30px;
                        margin: 0 auto;
                    }
                    .df-container {
                        width: 100%;
                        overflow-x: auto;
                        max-width: 8000px;
                        margin: 0 auto;
                        border-collapse: collapse;
                    }
                    .dataframe {
                        border: 0px;
                        width: 100%;
                        margin-bottom: 0;
                        border-collapse: collapse;
                        font-size:0.7rem;
                        line-height: 1.8; 
                        text-justify: inter-word;
                    }
                    .dataframe th, .dataframe td {
                        background-color:  #232229;
                        border-left: none;
                        border-right: none;
                        padding: 12px 8px;
                        vertical-align: top;
                        overflow: hidden;

                    }
                    .dataframe th {
                        border-color:#29292C;
                        border-left: none;
                        border-top: none;
                        border-right: none;

                        background-color: inherit;

                        font-weight: bold;
                        position: sticky;
                        top: 0;
                        z-index: 1;

                    }


                </style>


                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

                <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.css" integrity="sha384-D+9gmBxUQogRLqvARvNLmA9hS2x//eK1FhVb9PiU86gmcrBrJAQT8okdJ4LMp2uv" crossorigin="anonymous">

                <!-- The loading of KaTeX is deferred to speed up page rendering -->
                <script src="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/katex.min.js" integrity="sha384-483A6DwYfKeDa0Q52fJmxFXkcPCFfnXMoXblOkJ4JcA8zATN6Tm78UNL72AKk+0O" crossorigin="anonymous"></script>

                <!-- To automatically render math in text elements, include the auto-render extension: -->
                <script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.0-rc.1/dist/contrib/auto-render.min.js" integrity="sha384-yACMu8JWxKzSp/C1YV86pzGiQ/l1YUfE8oPuahJQxzehAjEt2GiQuy/BIvl9KyeF" crossorigin="anonymous"
                    onload="renderMathInElement(document.body);"></script>


            </head>
            <body>
                <div class="main">


                    <!-- TITLE START -->
                    <p style="text-align: center; color: #3F4456; font-size: 0.7rem; text-align: right;">Generated by LTA Datascience Team</p>
                    <h1>⚙&nbsp;RAG Evaluation Report</h1>
                    <br>
                    <h4> Evaluation Metric Scores</h4>
                    <!-- TITLE END -->


                    <!-- METRIC START -->
                    <div class="flex-row">
                         <div class="metric-box">

                            <div class="metric-title">ANSWER CORRECTNESS</div>

                            {% if ac is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value" style="color: 
                                    {% if ac > 70 %}#86A967{% else %}#EE6257{% endif %};">
                                    {{ ac | round(1) }}% </div>

                                <progress value={{ ac }} max=100></progress>
                            {% endif %}

                        </div>

                        <div class="metric-box">

                            <div class="metric-title">FAITHFULNESS</div>

                            {% if ff is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value" style="color: 
                                    {% if ff > 70 %}#86A967{% else %}#EE6257{% endif %};">
                                    {{ ff | round(2) }}% </div>

                                <progress value={{ ff }} max=100></progress>
                            {% endif %}

                        </div>

                        <div class="metric-box">

                            <div class="metric-title">ANSWER RELEVANCY</div>

                            {% if ar is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value" style="color: 
                                    {% if ar > 70 %}#86A967{% else %}#EE6257{% endif %};">
                                    {{ ar | round(2) }}% </div>

                                <progress value={{ ar }} max=100></progress>
                            {% endif %}

                        </div>

                        <div class="metric-box">

                            <div class="metric-title">CONTEXT RECALL</div>

                            {% if cr is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value" style="color: 
                                    {% if cr > 70 %}#86A967{% else %}#EE6257{% endif %};">
                                    {{ cr | round(2) }}% </div>

                                <progress value={{ cr }} max=100></progress>
                            {% endif %}

                        </div>

                        <div class="metric-box">

                            <div class="metric-title">CONTEXT PRECISION</div>

                            {% if cp is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value" style="color: 
                                    {% if cp >70 %}#86A967{% else %}#EE6257{% endif %};">
                                    {{ cp | round(2) }}% </div>

                                <progress value={{ cp }} max=100></progress>
                            {% endif %}

                        </div>


                    </div>

                    <div class="flex-row2">

                        <div class="metric-box2">

                            <div class="metric-title">BERT SCORE</div>

                            {% if bs is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value">{{ bs | round(2)  }}% </div>
                            {% endif %}

                        </div>

                        <div class="metric-box2">

                            <div class="metric-title">ROUGE SCORE</div>

                            {% if rg is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value">{{ rg | round(2) }}% </div>
                            {% endif %}

                        </div>

                        <div class="metric-box2">

                            <div class="metric-title">MRR</div>

                            {% if mrr is none %}
                                <div class="metric-none">UNAVAILABLE</div>
                            {% else %}
                                <div class="metric-value">{{ mrr | round(2) }}%</div>
                            {% endif %}

                        </div>

                    </div>
                    <!------ METRIC END ------->





                    <!------ TAB START ------->
                    <p style="text-align: left; font-weight: bold;">PERFORMANCE BY METRIC</p>

                    <!-- Tab links -->
                    <div class="tab">
                    {% if ac is not none %}
                      <button class="tablinks" onclick="openCity(event, 'Answer Correctness')">Answer Correctness</button>
                    {% endif %}

                    {% if ff is not none %}
                      <button class="tablinks" onclick="openCity(event, 'Faithfulness')">Faithfulness</button>
                    {% endif %}

                    {% if ar is not none %}
                      <button class="tablinks" onclick="openCity(event, 'Answer Relevancy')">Answer Relevancy</button>
                    {% endif %}

                    {% if cr is not none %}
                      <button class="tablinks" onclick="openCity(event, 'Context Recall')">Context Recall</button>
                    {% endif %}

                    {% if cp is not none %}
                      <button class="tablinks" onclick="openCity(event, 'Context Precision')">Context Precision</button>
                    {% endif %}

                    {% if bs is not none %}
                      <button class="tablinks" onclick="openCity(event, 'BERT')">BERT Score</button>
                    {% endif %}

                    {% if rg is not none %}
                      <button class="tablinks" onclick="openCity(event, 'ROUGE')">ROUGE Score</button>
                    {% endif %}

                    {% if mrr is not none %}
                      <button class="tablinks" onclick="openCity(event, 'MRR')">MRR</button>
                    {% endif %}

                    </div>

                    <!-- Tab content -->
                    <div id="Answer Correctness" class="tabcontent" style>

                        <h3 class="tabtext2">Component Evaluated: End-to-End</h3>

                        <h3 class="tabtext">Answer correctness evaluates the performance of the entire pipeline</h3>  

                        <div class="recommendation"><h3>Description</h3>
                            <p style="font-size: 0.97rem;">The answer correctness score evaluates the number factual statements and the semantic similarity between the answer and ground truth.
                            The factual accuracy is computed by the F1 score below, while semantic closeness is calculated with cosine similarity. These are then taken to calculate their average which is
                            the answer correctness score.</p>
                            
                            <br>
                            <p style = "color: #3F4456; font-size: 0.8rem;">$$\\text{F1 Score} = \\frac{|TP|}{(|TP| + 0.5 \\times (|FP| + |FN|))}$$</p>
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart1"></canvas>
                            <img src="my_kde_plot1.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>

                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> For factual accuracy, prompt engineering is important. Consider specific role prompting to the language model. 
                        Tuning temperature can also reduce the randomness of predictions and lead to more factual outputs.
                        Retriever parameters includes search indexes and re-ranking results may improve the factuality of retrieved context.</p>
                        </div>

                       <br><br><br>
                   </div>


                    <div id="Faithfulness" class="tabcontent">

                        <h3 class="tabtext2">Component Evaluated: LLM</h3>

                        <h3 class="tabtext">Faithfulness evaluates the performance of the large language model</h3>  


                        <div class="recommendation"><h3>Description</h3> 
                            <p style="font-size: 0.97rem;"> The faithfulness score measures how well the generated answer aligns with the provided context. 
                            To determine faithfulness, claims made in the answer are compared against the context and each claim is checked to see if it can be logically inferred from the provided context.</p> 
                            <br>
                            <p style = "color: #3F4456; font-size: 0.8rem;">$$\\text{Faithfulness} = \\frac{\\text{Number of claims in answer that can be inferred from context}}{\\text{Total number of claims in answer}}$$</p>
                        </div>

                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart2"></canvas>
                            <img src="my_kde_plot2.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>
                        
                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                    <br> <br> <br> </div>


                    <div id="Answer Relevancy" class="tabcontent">
                       
                        <h3 class="tabtext2">Component Evaluated: End-to-End</h3>
                        
                        <h3 class="tabtext">Answer relevancy evaluates the performance of the whole pipeline</h3>  

                        <div class="recommendation"><h3>Description</h3>   
                            <p class="tabtext" style="font-size: 0.97rem;"> The answer relevancy score focuses on assessing how relevant the generated answer is to the given prompt. 
                            Answer relevancy is computed by  the mean cosine similarity of the original question to a number of artifical questions which were reversed engineer from the answer. </p>
                        
                            <br>
    
                            <p style = "color: #3F4456; font-size: 0.8rem;">\[\\text{Answer Relevancy} = \\frac{1}{N} \sum_{i=1}^N \cos(E_{g_i}, E_o)\]</p>
                            <p style="font-size: 0.8rem;">Where:</p>
                            <ul style="font-size: 0.8rem;">
                                <li style = "text-align: left; line-height: 1.8;color: #3F4456;">Eg is the embedding of the generated question. </li>
                                <li style = "text-align: left; line-height: 1.8;color: #3F4456;">Eo is the embedding of the original question.</li>
                                <li style = "text-align: left; line-height: 1.8;color: #3F4456;">N is the number of generated questions, which is 3 by default.</li>
                            </ul> 
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart3"></canvas>
                            <img src="my_kde_plot3.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>
                        
                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>


                    <br> <br> <br></div>


                    <div id="Context Recall" class="tabcontent">
                    
                        <h3 class="tabtext2">Component Evaluated: Retriever</h3>
                        
                        <h3 class="tabtext">Context recall evaluates the performance of the retriever</h3>  

                        <div class="recommendation"><h3>Description</h3> 
                            <h3 class="tabtext">Description</h3>  
                            <p style="font-size: 0.97rem;">The context recall score evaluates how well the retrieved context may lead to inference of correct facts. 
                            This is computed by the number of sentences in the ground truth which can be inferred from the retrieved context.</p>
    
                            <br>
                            
                            <p style = "color: #3F4456; font-size: 0.8rem;">$$\\text{context recall} = \\frac{|GT \\text{ sentences that can be attributed to context}|}{|\\text{Number of sentences in GT}|}$$</p>
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart4"></canvas>
                            <img src="my_kde_plot4.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>
                        
                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                    <br> <br> <br></div>


                    <div id="Context Precision" class="tabcontent">

                        <h3 class="tabtext2">Component Evaluated: Retriever</h3>
                        
                        <h3 class="tabtext">Context Precision evaluates the performance of the retriever</h3> 

                        <div class="recommendation"><h3>Description</h3> 
                            <h3 class="tabtext">Description</h3>  
                            <p style="font-size: 0.97rem;">The context precision assesses whether all relevant items from the ground truth are ranked higher in the retrieved contexts. 
                            Each item in the top K ranked retrieved context is assesed to tell if it is relevant (TP) or not (FP). Context precision is calculated by the cumulative sum of precision values at each position over the total number of relevant items.</p>
    
                            <br><br><p style = "color: #3F4456; font-size: 0.8rem;">$$\\text{Context Precision@K} = \\frac{\\sum_{k=1}^K (\\text{Precision@k} \\times v_k)}{\\text{Total number of relevant items in the top K results}}$$</p>
    
                            <br><p style = "color: #3F4456; font-size: 0.8rem;"">$$\\text{Precision} = \\frac{True Positive@k}{True Positives@k + False Positives@k}$$</p>
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart5"></canvas>
                            <img src="my_kde_plot5.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>

                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                   <br> <br> <br> </div>

                    <div id="BERT" class="tabcontent">

                        <h3 class="tabtext2">Component Evaluated: End-to-End</h3>
                        
                        <h3 class="tabtext">Bert score evaluates the performance of the whole pipeline</h3> 

                        <div class="recommendation"><h3>Description</h3> 
                            <p style="font-size: 0.97rem;">BERTScore is an evaluation metric that assesses the similarity between a generated and reference text by leveraging contextual embeddings from the pre-trained Microsoft Deberta-xlarge-mnli. BERT Precision reflects the amount of unnecessary information included, while BERT Recall reflects how much necessary information is omitted. 
                            This provides a nuanced view on how sharp the model with regards to its knowledge base.</p>
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart6"></canvas>
                            <img src="my_kde_plot6.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>

                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                    <br> <br> <br></div>

                    <div id="ROUGE" class="tabcontent">
        
                        <h3 class="tabtext2">Component Evaluated: Large Language Model</h3>

                        <h3 class="tabtext">Rouge score evaluates the performance of the whole pipeline</h3> 

                        <div class="recommendation"><h3>Description</h3> 
                            <p style="font-size: 0.97rem;">BERTScore is an evaluation metric that assesses the similarity between a generated answer and a reference answer using contextual embeddings from BERT. It calculates a score based on how well the tokens and their order in the generated answer match those in the reference answer. Higher BERTScore values indicate greater similarity and alignment between the generated and reference answers.</p>
                        </div>

                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px; margin-right:15px; background-color: #272D44;" id="chart7"></canvas>
                            <img src="my_kde_plot7.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 8px;" >
                        </div>
        
                        <br>
                        
                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                    <br> <br> <br></div>

                    <div id="MRR" class="tabcontent">

                        <h3 class="tabtext2">Component Evaluated: Retriever</h3>

                        <h3 class="tabtext">Mean Reciprocal Ranking evaluates the performance of the retriever</h3> 

                        <div class="recommendation"><h3>Description</h3> 
                            <p style="font-size: 0.97rem;">Mean Reciprocal Ranking is a score that evaluates the ability of a retriever to prioritise relevant results. 
                            It is computed by measuring the cosine similarity between the retrieved contexts and the question, and then taking the reciprocal rank of the first relevant document.</p>
                            
                            <br>
    
                            <p style = "color: #3F4456;">$$\\text{Mean Reciprocal Ranking} = \\frac{1}{\\text{|Ranking @k of the first relevant document|}}$$</p>
                        </div>
                        
                        <br>
                        <div class="flex-row", style = "justify-content: center">
                            <canvas style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 10px; margin-right:15px; background-color: #272D44;" id="chart8"></canvas>
                            <img src="my_kde_plot8.png" style="width: 42%; box-shadow: 0 0 16px rgba(0, 0, 0, 0.8); border-radius: 10px;" >
                        </div>
        
                        <br>f

                        <div class="recommendation">
                        <div class="recommendation_text">RECOMMENDATION</div>
                        <p> Improve the performance of the Rewriter component, especially for conversational and double questions.
                        Additionally, focus on enhancing the retriever's handling of the "Others" topic to ensure balanced performance across all topics.</p>
                        </div>

                    <br> <br> <br></div>
                    <!------ TAB END ------->



                    <!------ TAB FUNC START------>
                    <script>
                    // Your provided script goes here
                    // Define the Utils object (since it's not standard in Chart.js)
                    const Utils = {
                        months: function({count}) {
                            const names = ['0.0<<0.05', '0.05<<0.1', '0.1<<0.15', '0.15<<0.2', '0.2<<0.25', '0.25<<0.3', 
                                          '0.3<<0.35', '0.35<<0.4', '0.4<<0.45', '0.45<<0.5', '0.5<<0.55', '0.55<<0.6',
                                          '0.6<<0.65', '0.65<<0.7', '0.7<<0.75', '0.75<<0.8', '0.8<<0.85', '0.85<<0.9',
                                          '0.9<<0.95', '0.95<<1.0']

                            return names.slice(0, count);
                        }
                    };

                    // Define colors once
                     const backgroundColors = [
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                    ];
                    const borderColors = [
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(255, 205, 86)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(128, 210, 148)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)',
                        'rgb(54, 162, 235)'
                    ];

                    // Function to create a chart
                    function createChart(canvasId, label, data) {
                        const labels = Utils.months({count: data.length});
                        const chartData = {
                            labels: labels,
                            datasets: [{
                                label: label,
                                data: data,
                                backgroundColor: backgroundColors,
                                borderColor: borderColors,
                                borderWidth: 1
                            }]
                        };

                        const config = {
                        type: 'bar',
                        data: chartData,
                        options: {
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            },
                            plugins: {
                                legend: {
                                    display: false  // This will hide the legend
                                },
                                title: {
                                    display: true,
                                    text: 'SCORE FREQUENCY DISTRIBUTION'
                                }
                            }
                        },
                    };

                        const ctx = document.getElementById(canvasId).getContext('2d');
                        new Chart(ctx, config);
                    }

                    // Usage
                    createChart('chart1', 'SCORE FREQUENCY DISTRIBUTION', {{ ac_dist }});
                    createChart('chart2', 'SCORE FREQUENCY DISTRIBUTION', {{ ff_dist }});
                    createChart('chart3', 'SCORE FREQUENCY DISTRIBUTION', {{ ar_dist }});
                    createChart('chart4', 'SCORE FREQUENCY DISTRIBUTION', {{ cr_dist }});
                    createChart('chart5', 'SCORE FREQUENCY DISTRIBUTION', {{ cp_dist }});
                    createChart('chart6', 'SCORE FREQUENCY DISTRIBUTION', {{ bs_dist }});
                    createChart('chart7', 'SCORE FREQUENCY DISTRIBUTION', {{ rg_dist }});
                    createChart('chart8', 'SCORE FREQUENCY DISTRIBUTION', {{ mrr_dist }});
                    </script>

                    <script>
                    function openCity(evt, cityName) {
                      // Declare all variables
                      var i, tabcontent, tablinks;

                      // Get all elements with class="tabcontent" and hide them
                      tabcontent = document.getElementsByClassName("tabcontent");
                      for (i = 0; i < tabcontent.length; i++) {
                        tabcontent[i].style.display = "none";
                      }

                      // Get all elements with class="tablinks" and remove the class "active"
                      tablinks = document.getElementsByClassName("tablinks");
                      for (i = 0; i < tablinks.length; i++) {
                        tablinks[i].className = tablinks[i].className.replace(" active", "");
                      }

                      // Show the current tab, and add an "active" class to the button that opened the tab
                      document.getElementById(cityName).style.display = "block";
                      evt.currentTarget.className += " active";
                    }
                    </script>
                    <!------ TAB FUNC END------>

                    <br><br>

                    <!------ TABLE START ------>
                    <h4> Compiled Dataset </h4>
                    <hr> <!-- Horizontal line -->

                </div>
                     <!------ Generate checkboxes based on DataFrame columns ------>
                    <div>
                        {% for col in df.columns %}
                            <label><input type="checkbox" class="toggle-column" data-column="{{ loop.index }}" checked> {{ col }}</label>
                        {% endfor %}
                    </div>
                    <br><br><br><br><br>

                    <div class="table-container">
                        {{ df_html|safe }}
                    </div>
                     <!------ TABLE END------>

                     <script>
                        document.addEventListener('DOMContentLoaded', function() {
                            document.querySelectorAll('.toggle-column').forEach(function(checkbox) {
                                checkbox.addEventListener('change', function() {
                                    var column = this.getAttribute('data-column');
                                    var index = parseInt(column) - 1;
                                    var table = document.querySelector('.dataframe');
                                    var rows = table.rows;

                                    for (var i = 0; i < rows.length; i++) {
                                        var cells = rows[i].cells;
                                        if (cells[index]) {
                                            cells[index].style.display = this.checked ? '' : 'none';
                                        }
                                    }
                                });
                            });
                        });
                    </script>

                </div>
            </body>
            </html>
            """