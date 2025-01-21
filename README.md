# King's Capital Quant Division
Part of the King's Capital paper: *Sentiment analysis on equities using Large Language Models*


## Financial News Sentiment Analyzer

A Python-based sentiment analysis tool that evaluates market sentiment through financial news headlines and quarterly reports using FinBERT. Aggregates data from multiple sources to provide comprehensive sentiment insights.


## Features

- Financial sentiment analysis using FinBERT model
- Multi-source news aggregation (CNBC, Yahoo Finance, Quarterly Reports)
- Interactive data visualization
- Configurable analysis parameters
- Support for any stock symbol*

*Quarterly Review analysis currently limited to AAPL


## Dependencies

```
transformers
torch
pandas
numpy
requests
beautifulsoup4
yfinance
matplotlib
seaborn
```

## Usage

Basic usage (defaults to AAPL stock):
```python
python main.py
```

To analyze a different stock, modify `Config` class in the code:
```python
class Config:
    stock_symbol = 'MSFT'  # Change to your desired stock symbol
```

## Configuration

The `Config` class allows customization of:

```python
class Config:
    stock_symbol = 'AAPL'
    MIN_HEADLINE_LENGTH = 5
    SENTIMENT_WEIGHTS = {
        'positive': 1.0,
        'neutral': 0.0,
        'negative': -1.0
    }
    USE_CNBC = True
    USE_YF = True
    USE_QUARTERLY_REVIEW = True
```

## Output

The analyzer provides:
- Overall sentiment score (-100% to 100%)
- Visual sentiment distribution
- Detailed sentiment breakdown
- Summary statistics


## Limitations

- Quarterly review analysis currently only works for Apple (AAPL)
- Subject to API rate limits
- Web scraping may break if source websites change their structure
