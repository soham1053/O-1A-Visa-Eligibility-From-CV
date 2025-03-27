# Design

## Modularity
Separated the tasks for classifying the CV into independent parts for horizontal development:
1. Extracting the text from the CV
2. Parsing the knowledge base of O-1 criteria
    * I used simple RAG to chunk the O-1 criteria from USCIS with the `all-MiniLM-L6-v2` sentence transformer. In this exercise I chunked a small part of the USCIS-provided info, but later one could instead just scrape the relevant pages to stay updated and have access to all the information.
3. Extract evidence of possibly met criteria
    * Used a naive approach to segment the CV into section-context pairs, by simply prompting `gpt-4o` to respond in JSON format.
    * I used `bart-large-mnli` to do zero-shot classification for evidence categorization based on each segment. Given its smaller size, could be more easily fine-tuned? 
4. Applying domain knowledge of specific field the person is in, if available
    * For this exercise, I use a simple implementation of field domain knowledge by creating a dictionary that contains keywords for each field and criterion.
5. Computing the criterion and overall ratings
    * I weighted all the criteria with different importance, these can be tuned(also the classifier threshold), possibly formulating the problem as an RL bandit.

These components can be iterated and eventually integrate well with each other, which they don't right now due to time constraints.

---

# Output Evaluation

First, install the required packages from `requirements.txt`. You might have to run `import nltk; nltk.download('punkt_tab')` 
if a related error appears. 

Run the app with `uvicorn main:app --reload`, open `http://127.0.0.1:8000/docs`, and run `evaluate`, inputting a field(e.g. Computer Science) 
and a CV(`.pdf`). 

You'll get:
```
{
for_each_criterion : {
   text_segment_of_cv : "Worked at ...",
   classifier_confidence : 0.7675, 
   section_of_cv : "Experience",
   boosted : True (if specific keywords are found in CV, score is multiplied), 
   uscis_reference : "The supporting documentation for an O-1A petition must include..." (from RAG)
   }
overall_rating : "Medium",
score : 3.2 (to be more precise than overall), 
criterion_scores : scores_per_criterion, 
field : "Computer Science"
}
```