import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("water_access_data.csv")


group1 = ["India", "Zimbabwe", "Costa Rica"]
group2 = ["United States of America", "Azerbaijan", "Argentina"]



def bar_graph(country):
    plt.figure(figsize=(10, 6))
    plt.bar(df['Years'], df[country], color='purple', alpha=0.7, label=country)
    plt.title(f'% of people who have access to drinking water in {country} over the years')
    plt.xlabel('Years')
    plt.ylabel('Percentage')
    plt.legend()
    plt.grid(axis='y')
    plt.show()



def line_graph(countries):
    plt.figure(figsize=(10, 6))
    for country in countries:
        plt.plot(df['Years'], df[country], label=country)
    plt.title('Access to Drinking Water over the Years')
    plt.xlabel('Years')
    plt.ylabel('% of people with access')
    plt.legend()
    plt.grid()
    plt.show()



def heatmap(countries):
    selected_data = df[countries]
    sns.heatmap(selected_data.corr(), annot=True, cmap='viridis', fmt='.3f', linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()



def scatter_plot():
    melted_data = df.melt(id_vars='Years', var_name='Country', value_name='Percentage')
    sns.set_theme()
    sns.scatterplot(data=melted_data, x='Country', y='Percentage', hue='Years', palette='viridis', s=50)
    plt.title('Scatter Plot of Percentages by Country and Year')
    plt.xticks(rotation=45)
    plt.show()




def menu():
    while True:
        print("\nMenu:")
        print("1. Bar Graph for a Country")
        print("2. Line Graph for Selected Countries")
        print("3. Heatmap for Selected Countries")
        print("4. Scatter Plot for All Countries")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            country = input("Enter the country: ")
            if country in df.columns:
                bar_graph(country)
            else:
                print("Country not found in the dataset!")
        elif choice == "2":
            countries = input("Enter the countries separated by commas: ").split(',')
            countries = [c.strip() for c in countries]
            if all(c in df.columns for c in countries):
                line_graph(countries)
            else:
                print("One or more countries not found in the dataset!")
        elif choice == "3":
            countries = input("Enter the countries separated by commas: ").split(',')
            countries = [c.strip() for c in countries]
            if all(c in df.columns for c in countries):
                heatmap(countries)
            else:
                print("One or more countries not found in the dataset!")
        elif choice == "4":
            scatter_plot()
        elif choice == "5":
            print("Exiting the program.")
            break
        else:
            print("Invalid choice! Please try again.")



menu()

