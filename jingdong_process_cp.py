import time
import sys
import random
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup as bs
# from multiprocessing import Pool
# 引入配置对象DesiredCapabilities
# from selenium.webdriver.common.desired_capabilities import DesiredCapabilities


class jingDong(object):

    def __init__(self):
        super(jingDong, self).__init__()
        options = webdriver.ChromeOptions()
        options.add_argument('disable-infobars')
        # 设置无图模式
        prefs = {'profile.default_content_setting_values': {'images': 2}}
        options.add_experimental_option('prefs', prefs)
        # 窗口最大化
        options.add_argument('--kiosk')
        self.driver = webdriver.Chrome(chrome_options=options)

        # 页码标识
        self.page_num = 1
        # 评论数标识
        self.comment_num = 0
        self.comment_list = []
        self.index_dict = {}
        self.star_total = {}
        self.itemInit()
        self.loadPage()

    def itemInit(self):
        while True:
            item_index = random.randint(1000000, 9999999)
            if item_index not in self.index_dict:
                break
        self.index_dict[item_index] = 1
        item_url = 'https://item.jd.com/%s.html#comment' % str(item_index)
        self.driver.get(item_url)
        print(item_url)

    def loadPage(self):
        self.no_comment_flag = 0
        while True:
            try:
                # time.sleep(1)
                # chrome的问题，如果点击元素不在窗口就会报错，显示这个元素不可点击
                self.driver.execute_script(
                    "window.scrollBy(150,300)", "")  # 向下滚动200px
                path = "//*[@id='comment-0']//a[@clstag='shangpin|keycount|product|pinglunfanye-nextpage']"
                element = WebDriverWait(self.driver, 10).until(EC.presence_of_element_located(
                    (By.XPATH, path)))
            except NoSuchElementException as e:
                if self.page_num == 5000000:
                    break
                else:
                    self.itemInit()
                    continue
            except:
                # 用来判断有无评论
                if self.no_comment_flag <= 2:
                    time.sleep(1)
                    self.no_comment_flag += 1
                else:
                    self.itemInit()
                continue
            soup = bs(self.driver.page_source, 'lxml')
            comments = soup.find('div', id='comment-0').find_all(
                'div', class_='comment-column J-comment-column')
            # 提取出评论的等级以及评论的内容
            for item in comments:
                comment_level = item.find('div')['class']
                comment_text = item.find('p', {'class': 'comment-con'})
                self.comment_list.append(
                    (comment_level, comment_text.get_text()))
            print('%d page is OK!' % (self.page_num))
            if self.page_num % 100 == 0:
                self.saveFile()
            self.page_num += 1
            try:
                element.click()
            except Exception as e:
                self.itemInit()
            else:
                pass
            # time.sleep(2)
            # self.driver.save_screenshot('jingdong.png')
        print('Comment total : %d ' % (self.comment_num))
        with open('comment_score.txt', 'a') as f:
            for star, num in self.star_total.items():
                f.write('star' + star + ':' + num + '\n')
        # self.driver.save_screenshot('/home/python3/python/jingdong.png')
        self.driver.quit()

    def saveFile(self):
        '''保存评论的等级以及评论的内容'''
        self.comment_num += 1
        with open('comment_score.txt', 'a') as f:
            for i in range(len(self.comment_list)):
                if self.comment_list[i][0][1][-1] not in self.star_total:
                    self.star_total[self.comment_list[i][0][1][-1]] = 0
                self.star_total[self.comment_list[i][0][1][-1]] += 1
                f.write(self.comment_list[i][0][1][-1] +
                        '\t' + self.comment_list[i][1] + '\n')
        with open('comment.txt', 'a') as f:
            for i in range(len(self.comment_list)):
                f.write(self.comment_list[i][1] + '\n')
        self.comment_list = []
        print(self.star_total)


def itemListInit():
    '''根据输入的商品名和等级构成元祖，并加入到列表中，以便对多个商品进行处理'''
    item_list = []
    while True:
        print('Please enter the item infomation or enter q/Q quit system!')
        item = input('Please enter the item you want to inquire about:')
        # 如果没有输入回车了，则认为输入完成，退出输入，开始执行爬虫
        if len(item) == 0:
            break
        # 如果输入q或Q则直接退出系统
        elif item in ['q', 'Q']:
            sys.quit()
        level = input(
            'Please enter the item comment level you want to inquire about(1-5):')
        if level.isdigit():
            level = int(level)
        else:
            print('Enter is Error!')
            continue
        item_list.append((item, level))
    return item_list


def main():
    # item_list = itemListInit()
    # po = Pool(5)
    # for item in item_list:
    #     po.apply_async(target=jingDong, args=item)
    # print('------start------')
    # po.close()
    # po.join()
    # print('------end------')
    jingdong = jingDong()


if __name__ == '__main__':
    main()
