# US-baby-names

Data from https://www.kaggle.com/kaggle/us-baby-names


## Example usage

Compare multiple names:

	python analysis.py --names Wells Howard Jamie Louis Darren --gender=M -o namePlot_compare.png

Normalize and compare shapes of names:

	python analysis.py --names Darren Daren Darin --gender=M -o namePlot_Darrens_norm.png --normalize

Plot single name and year when individual with that name was born:

	python analysis.py --names=Darren --year=1982 --gender=M

Get help menu with option arguments:

	python analysis.py --help
