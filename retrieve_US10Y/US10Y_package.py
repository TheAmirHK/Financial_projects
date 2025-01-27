from bs4 import BeautifulSoup
import requests

def get_US10Y_rate():
    url = "https://www.cnbc.com/quotes/US10Y"
    html_content = requests.get(url).text
    soup = BeautifulSoup(html_content, "lxml")
    head = soup.find('head')
    head = str(head)
    positions = 1173086  # Replace with the specific position you're interested in
    rate = ""
    for position in range(positions, positions + 4):  # Range from 3 to 6 (inclusive)
        if position < len(head):
            specific_value = head[position]
            rate += specific_value
    return float(rate)