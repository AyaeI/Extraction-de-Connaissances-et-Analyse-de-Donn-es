import pandas as pd
from tabulate import tabulate
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Function to analyze association rules
def association_rules_analysis(X):
    X_bool = X.applymap(lambda x: True if x > 0 else False)
    frequent_itemsets = apriori(X_bool, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.9)
    rules['count'] = rules['support'] * len(X)  # Calculate the count based on support
    return rules[['antecedents', 'consequents', 'confidence', 'support', 'lift', 'count']]

# Function to filter non-redundant rules
def filter_non_redundant_rules(rules):
    # Convert lists to tuples to allow duplicate removal
    rules['antecedents_tuple'] = rules['antecedents'].apply(tuple)
    rules['consequents_tuple'] = rules['consequents'].apply(tuple)
    
    # Remove duplicates based on antecedent and consequent tuples
    rules_no_redundant = rules.sort_values(by='lift', ascending=False).drop_duplicates(subset=['antecedents_tuple', 'consequents_tuple'])
    
    # Return non-redundant rules
    return rules_no_redundant

# Execution of the analysis
if __name__ == "__main__":
    # Load the dataset
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
    ]
    df = pd.read_csv('adult/cleanadult.data', header=None, names=columns)

    # Convert dataframe to list of transactions
    transactions = df.astype(str).values.tolist()

    # Apply TransactionEncoder
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

    print("Association Rules Analysis")
    rules = association_rules_analysis(df_encoded)

    # Convert frozensets to lists to remove frozenset type
    rules['antecedents'] = rules['antecedents'].apply(list)
    rules['consequents'] = rules['consequents'].apply(list)

    # Save all association rules to a text file
    all_rules_file_path = r"./Association_Rules/AR.txt"
    with open(all_rules_file_path, "w") as file:
        file.write("All Association Rules:\n")
        file.write(tabulate(rules[['antecedents', 'consequents', 'confidence', 'support', 'lift', 'count']], headers='keys', tablefmt='psql'))

    # Filter rules with confidence of 1
    rules_confidence_1 = rules[rules['confidence'] == 1]

    # Convert frozensets to lists to remove frozenset type
    rules_confidence_1['antecedents'] = rules_confidence_1['antecedents'].apply(list)
    rules_confidence_1['consequents'] = rules_confidence_1['consequents'].apply(list)

    # Convert DataFrame to a string in tabular format
    table_string = tabulate(rules_confidence_1[['antecedents', 'consequents', 'confidence',
                                                'support', 'lift', 'count']], headers='keys', tablefmt='psql')

    # Save the string of the table to a text file
    confidence_1_rules_file_path = r"./Association_Rules/AR_C=1.txt"
    with open(confidence_1_rules_file_path, "w") as file:
        file.write("Association Rules with Confidence 1:\n")
        file.write(table_string)

    print("All association rules are saved in:", all_rules_file_path)
    print("Association rules with confidence 1 are saved in:", confidence_1_rules_file_path)

    # Filter non-redundant rules and save them
    non_redundant_rules = filter_non_redundant_rules(rules)

    # Convert frozensets to lists to remove frozenset type
    non_redundant_rules['antecedents'] = non_redundant_rules['antecedents'].apply(list)
    non_redundant_rules['consequents'] = non_redundant_rules['consequents'].apply(list)

    # Convert DataFrame to a string in tabular format
    non_redundant_table_string = tabulate(non_redundant_rules[['antecedents', 'consequents', 'confidence', 
                                                               'support', 'lift', 'count']], headers='keys', tablefmt='psql')

    # Save the string of the table to a text file
    non_redundant_rules_file_path = r"./Association_Rules/AR_Non_Redundant.txt"
    with open(non_redundant_rules_file_path, "w") as file:
        file.write("Non-Redundant Association Rules:\n")
        file.write(non_redundant_table_string)

    print("Non-redundant association rules are saved in:", non_redundant_rules_file_path)
