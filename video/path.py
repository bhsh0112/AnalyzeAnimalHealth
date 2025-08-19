from datetime import datetime, timedelta

locations = ['twtrainer', 'twtrch']     # twtrainwe - 沙田; twtrch - 从化
trainers = ['ccw', 'cas', 'edj', 'fc', 'had', 'hda', 'lfc', 'lkw', 'mkl', 'mwk', 'nm', 'npc', 'rw', 'scs', 'sjj', 'swy', 'tkh', 'tys', 'wdj', 'ych', 'ypf', 'ytp']
VID_RES = '1200kbps'
duration = ['20200311', '20250320']     # ['start_date', 'end_date']

def date_range():
    start = datetime.strptime(duration[0], '%Y%m%d')
    end = datetime.strptime(duration[1], '%Y%m%d')
    cur = start

    while cur <= end:
        yield cur.strftime('%Y%m%d')
        cur += timedelta(days=1)

def gen_url(output_file):
    with open(output_file, 'w') as f:
        for date in date_range():
            for location in locations:
                for trainer in trainers:
                    url = f'https://streaminghkjc-a.akamaihd.net/hdflash/{location}/{date[:4]}/{date}/{trainer}/chi/{location}_{date}_{trainer}_chi_{VID_RES}.mp4\n'
                    f.write(url)

if __name__ == '__main__':
    gen_url("vid.txt")
