from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

ARTICLE = """ After her husband's assassination and funeral in 1963, Kennedy and her children largely withdrew from public view. In 1968, she married Greek shipping magnate Aristotle Onassis, which caused controversy. Following Onassis's death in 1975, she had a career as a book editor in New York City, first at Viking Press and then at Doubleday, and worked to restore her public image. Even after her death, she ranks as one of the most popular and recognizable first ladies in American history, and in 1999, she was listed as one of Gallup's Most-Admired Men and Women of the 20th century.[5] She died in 1994 and was buried at Arlington National Cemetery alongside President Kennedy and two of their children, one stillborn and one who died shortly after birth.[6] Surveys of historians conducted periodically by the Siena College Research Institute since 1982 have consistently found Kennedy Onassis to rank among the most highly regarded first ladies by the assessments of historians.
"""
text = summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False)

for item in text:
    for value in item.values():
        print(value)
