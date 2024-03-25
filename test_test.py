import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


def testando():
  driver = webdriver.Chrome()
  valores = []
  cards = ["gaea's cradle"]
  for cards in cards:
    driver.get("https://www.ligamagic.com.br/?view=cards/card&card="+f'{cards}')
    driver.set_window_size(1054, 800)
    valor = driver.find_element(By.CLASS_NAME, "col-prc-menor")
    valor = valor.get_attribute('innerText')
    valores = valor
    print('o valor da carta é: ')
    print(valor)
    driver.quit()
  
  print(valores)

if __name__ == '__main__':
    testando()