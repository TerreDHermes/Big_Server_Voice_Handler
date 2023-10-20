import subprocess
from pydub import AudioSegment
import os
import speech_recognition as sr
import codecs
import torch
from collections import Counter
import nltk
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import fonts
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = WordNetLemmatizer()
from pymorphy3 import MorphAnalyzer
from textblob import TextBlob
import time


# Функция для записи пути обработанного файла в список
def mark_file_as_processed(file_path, processed_file_set):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(file_path + '\n')
        file.flush()
    processed_file_set.add(file_path)

# Функция для чтения списка обработанных файлов
def read_processed_files(file_path):
    processed_files = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                processed_files.add(line.strip())
                file.flush()
    return processed_files

custom_stopwords = [
    "как бы", "собственно говоря", "таким образом", "буквально", "прямо",
    "как говорится", "так далее", "скажем", "ведь", "как его", "в натуре",
    "так вот", "короче", "как сказать", "видишь", "слышишь", "типа",
    "на самом деле", "вообще", "в общем-то", "в общем", "в некотором роде",
    "на фиг", "на хрен", "в принципе", "итак", "типа того", "только",
    "вот", "в самом деле", "всё такое", "в целом", "то есть",
    "это", "это самое", "ешкин кот", "ну", "ну вот", "ну это", "прикинь",
    "прикол", "значит", "знаешь", "так сказать", "понимаешь", "допустим",
    "слушай", "например", "просто", "конкретно", "блин",
    "походу", "а-а-а", "э-э-э", "не вопрос", "без проблем", "практически",
    "фактически", "как-то так", "ничего себе", "достаточно"
]
Mate_Words = [
    "Апездал", "Апездошенная", "Блядь", "Блядство", "Выебон", "Выебать", "Вхуюжить", "Гомосек",
    "Долбоёб","Ебло","Ебаный","Ёбаный","Еблище","Ебать","Ебическая сила","Ебунок","Еблан","Ёбнуть","Ёболызнуть",
    "Ебош","Заебал","Заебатый","Злаебучий","Заёб","Иди на хуй","Колдоебина","Манда","Мандовошка","Мокрощелка",
    "Наебка","Наебал","Наебаловка","Напиздеть","Отъебись","Охуеть","Отхуевертить","Опизденеть","Охуевший","Отебукать","Пизда","Пидарас",
    "Пиздатый","Пиздец", "Пизданутый","Поебать", "Поебустика","Проебать","Подзалупный","Пиздены","Припиздак","Разъебать","Распиздяй","Разъебанный","Сука","Сучка","Трахать","Уебок","Уебать",
"Угондошить","Уебан","Хитровыебанный","Хуй","Хуйня","Хуета","Хуево","Хуесос","Хуеть","Хуевертить","Хуеглот","Хуистика","Членосос","Членоплет","Шлюха"]


def analyze_sentiment(text):
    # Преобразуем текст в объект TextBlob
    analysis = TextBlob(text)

    # Определяем тональность текста
    if analysis.sentiment.polarity > 0:
        return "Настроение в тексте: позитивное"
    elif analysis.sentiment.polarity < 0:
        return "Настроение в тексте: негативное"
    else:
        return "Настроение в тексте: нейтральное"


def get_top_words_with_percentage(text, n=5):
    # Разбиваем текст на слова
    words = text.split()

    # Подсчитываем частоту употребления каждого слова
    word_counts = Counter(words)

    # Выбираем топ N слов
    top_words = word_counts.most_common(n)

    # Рассчитываем процентное содержание для каждого слова
    total_word_count = len(words)
    top_words_with_percentage = [(word, count, (count / total_word_count) * 100) for word, count in top_words]

    # Формируем строку с результатами
    top_words_str = "Топ {} слов: ".format(n)
    for word, count, percentage in top_words_with_percentage:
        top_words_str += f"{word}({percentage:.2f}%), "

    # Удаляем последнюю запятую и пробел
    top_words_str = top_words_str.rstrip(', ')

    return top_words_str


def mate(text, mate_words):
    # Преобразовываем текст в нижний регистр
    text = text.lower()

    # Инициализируем список матных слов и их процентного содержания
    mate_info = []

    # Создаем экземпляр pymorphy2
    morph = MorphAnalyzer()

    # Токенизируем текст, разбив его на слова
    words = word_tokenize(text)
    print(words)
    # Используем pymorphy2 для лемматизации слов
    lemmatized_words = [morph.parse(word)[0].normal_form for word in words]
    print(lemmatized_words)

    # Используем регулярное выражение для поиска матных слов в тексте
    for word in mate_words:
        pattern = r'\b' + re.escape(word) + r'\b'  # Используем \b для поиска только целых слов
        mate_count = lemmatized_words.count(word)

        # Если слово было найдено в тексте, добавляем его в список
        if mate_count > 0:
            # Рассчитываем процент матных слов
            total_word_count = len(lemmatized_words)
            if total_word_count > 0:
                mate_percentage = (mate_count / total_word_count) * 100
            else:
                mate_percentage = 0.0

            # Добавляем слово и его процентное содержание в список
            mate_info.append(f"{word}({mate_percentage:.2f}%)")

    # Составляем строку с информацией о матных словах, разделяя их запятыми
    mat_info = "Матные слова: " + " , ".join(mate_info)
    return mat_info

def count_stopwords(text, custom_stopwords):
    words = text.split()

    # Подсчитываем общее количество слов
    total_word_count = len(words)

    # Подсчитываем слова-паразиты
    stopwords_count = Counter([word.lower() for word in words if word.lower() in custom_stopwords])

    # Создаем строку с информацией о словах-паразитах и их процентном содержании
    stopwords_info = "Слова паразиты: "
    for word, count in stopwords_count.items():
        percentage = (count / total_word_count) * 100
        stopwords_info += f"{word}({percentage:.2f}%) , "

    # Удаляем последнюю запятую и пробел
    stopwords_info = stopwords_info.rstrip(', ')

    return stopwords_info


def get_top_words(text, n=10):
    words = text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(n)
    return top_words


def text_to_pdf(text_file_path, stopwords_info, mate_words, result , sentiment,audio_filename_path):
    # Определите путь и имя для PDF-файла
    pdf_file_path = text_file_path.replace('.txt', '.pdf')

    # Создаем PDF-файл
    doc = SimpleDocTemplate(pdf_file_path, pagesize=letter)

    # Создаем список элементов для PDF
    elements = []

    # Открываем файл .txt для чтения текста
    with open(audio_filename_path, 'r', encoding='utf-8') as text_file:
        text = text_file.read()
    pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
    # Определяем стиль для текста
    styles = getSampleStyleSheet()
    styleN = styles['Normal']

    # Укажите явно шрифт, поддерживающий кириллицу
    font_name = 'Arial'  # Пример шрифта, поддерживающего кириллицу
    styleN.fontName = font_name

    # Создаем элемент Paragraph с текстом
    elements.append(Paragraph(text, styleN))

    # Добавляем разделительную черту
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(stopwords_info, styleN))

    # Добавляем разделительную черту
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(mate_words, styleN))

    # Добавляем разделительную черту
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(result, styleN))

    # Добавляем разделительную черту
    elements.append(Spacer(1, 12))
    elements.append(HRFlowable(width="100%"))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(sentiment, styleN))


    # Собираем PDF
    doc.build(elements)

    # Возвращаем путь к созданному PDF-файлу
    return pdf_file_path

def read_text_from_file(text_path):
    with codecs.open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def add_punctuation_to_text(text_path):
    model, example_texts, languages, punct, apply_te = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                                                      model='silero_te')
    with codecs.open(text_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.lower()
    output_text = apply_te(text, lan='ru')

    with codecs.open(text_path, 'w', encoding='utf-8') as file:
        file.write(output_text)

    return output_text


def audio_to_text(audio_path, text_path):
    # Read the audio file
    try:
        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data, language='ru-RU')

        # Write the result to a text file
        os.makedirs(os.path.dirname(text_path), exist_ok=True)
        with open(text_path, 'w', encoding="utf-8") as f:
            f.write(text)

        os.remove(audio_path)
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
        return None

def convert_3gp_to_wav(input_path, output_path):
    try:
        # Выполняем команду FFmpeg для конвертации 3gp в WAV
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
            check=True)
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при конвертации: {e}")
        return None

    # Используем PyDub для загрузки и обработки WAV
    audio = AudioSegment.from_wav(output_path)
    return audio

def create_all_text_file(text_path, text_directory):
    # Создаем путь к файлу ALL_TEXT.txt
    all_text_path = os.path.join(text_directory, "ALL_TEXT.txt")

    # Открываем файл ALL_TEXT.txt для добавления текста (режим 'a')
    with open(all_text_path, 'a', encoding='utf-8') as all_text_file:
        # Открываем файл text_path для чтения текста
        with open(text_path, 'r', encoding='utf-8') as text_file:
            # Читаем текст из text_path и записываем его в ALL_TEXT.txt
            text = text_file.read()
            all_text_file.write(text)

    print(f"Text appended to {all_text_path}")
    return all_text_path

def process_emails(base_directory, processed_file_set, proverka_path):
    for root, dirs, files in os.walk(base_directory):
        if "audio" in dirs:
            email_directory = root
            audio_directory = os.path.join(email_directory, "audio")
            text_directory = os.path.join(email_directory, "text")

            if not os.path.exists(text_directory):
                os.makedirs(text_directory, exist_ok=True)

            for filename in os.listdir(audio_directory):
                if filename.lower().endswith(".3gp"):
                    audio_path = os.path.join(audio_directory, filename)
                    if audio_path not in processed_file_set:
                        processed_file_set.add(audio_path)
                        # Записываем обновленный список обработанных файлов в файл base.txt
                        with open(proverka_path, 'w') as base_file:
                            base_file.write('\n'.join(processed_file_set))

                        text_filename = os.path.splitext(filename)[0] + '.txt'
                        text_path = os.path.join(text_directory, text_filename)

                        audio_filename = os.path.splitext(filename)[0] + '.txt'
                        audio_filename_path = os.path.join(audio_directory, audio_filename)

                        if not os.path.exists(text_path):
                            print(f"Processing {audio_path}...")
                            convert_3gp_to_wav(audio_path, audio_filename_path.replace(".txt", ".wav"))
                            try:
                                audio_to_text(audio_filename_path.replace(".txt", ".wav"), audio_filename_path)
                            except Exception as e:
                                time.sleep(5)
                                audio_to_text(audio_filename_path.replace(".txt", ".wav"), audio_filename_path)
                            #audio_to_text(audio_filename_path.replace(".txt", ".wav"), audio_filename_path)
                            output_text = add_punctuation_to_text(audio_filename_path)
                            all_text_path = create_all_text_file(audio_filename_path, audio_directory)
                            output_text2 = read_text_from_file(all_text_path)

                            # output_text = [word.lower() for word in output_text]
                            # Вызов функции для выявления слов-паразитов
                            Mate_Words2 = [word.lower() for word in Mate_Words]
                            stopwords = count_stopwords(output_text, custom_stopwords)
                            mate_words = mate(output_text, Mate_Words2)
                            result = get_top_words_with_percentage(output_text, n=10)
                            sentiment = analyze_sentiment(output_text)

                            stopwords2 = count_stopwords(output_text2, custom_stopwords)
                            mate_words2 = mate(output_text2, Mate_Words2)
                            result2 = get_top_words_with_percentage(output_text2, n=10)
                            sentiment2 = analyze_sentiment(output_text2)

                            # pdf_filename = os.path.splitext(filename)[0] + '.txt'
                            # text_path = os.path.join(text_directory, text_filename)
                            filename = os.path.basename(all_text_path)
                            pdf_path = os.path.join(text_directory, filename)

                            text_path_pdf = text_to_pdf(text_path, stopwords, mate_words, result, sentiment, audio_filename_path)
                            text_path_pdf2 = text_to_pdf(pdf_path, stopwords2, mate_words2, result2, sentiment2, all_text_path)
                            #mark_file_as_processed(audio_path, processed_file_set)
                            # Подождите некоторое время, чтобы убедиться, что другие процессы завершили использование файла
                            time.sleep(2)

                            try:
                                os.remove(audio_filename_path)
                            except Exception as e:
                                time.sleep(5)
                                os.remove(audio_filename_path)

                            print(f"Text saved to {text_path_pdf}")
                            print(f"Text saved to {text_path_pdf2}")



def main():
    file_path = 'C:/base/Prroverka.txt'
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            file.flush()
        print(f"Создан файл {file_path}.")
    base_directory = '/base'
    # Читаем содержимое файла base.txt для получения списка обработанных файлов
    with open(file_path, 'r') as base_file:
        processed_file_set = set(base_file.read().splitlines())
    #processed_file_set = read_processed_files(file_path)
    print(processed_file_set)
    while(True):
        process_emails(base_directory, processed_file_set,file_path)

if __name__ == '__main__':
    main()

