""" 
PDF Splitter using ML
"""

import argparse
import asyncio
import json
import os
import pickle
import re

import cv2
import numpy as np
from pdf2image import convert_from_path
from pdf2image.pdf2image import pdfinfo_from_path
from PyPDF2 import PdfReader, PdfWriter
from tqdm import tqdm


def extract_image_features(pdf_path, page_number, target_size=(128, 128)):
    """
    Extracts visual features from a PDF page by converting it to an image,
    resizing it, and returning the pixel data as a NumPy array.

    Args:
        pdf_path (str): Path to the PDF file.
        page_number (int): Page number to extract features from (1-indexed).
        target_size (tuple): Desired size of the image (width, height).

    Returns:
        numpy.array: Resized and preprocessed image data as a NumPy array,
                     or None if an error occurs.
    """
    try:
        images = convert_from_path(
            pdf_path, first_page=page_number, last_page=page_number, dpi=100)
        if images:
            pil_image = images[0]

            # Convert PIL Image to NumPy array
            cv_image = np.array(pil_image)
            # OpenCV expects BGR format.
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # Resize the image for consistent input to the CNN
            resized_image = cv2.resize(cv_image, target_size)

            # Normalize pixel values to be between 0 and 1
            normalized_image = resized_image / 255.0

            return normalized_image
        else:
            return None
    except Exception as e:
        print(
            f"Error extracting image features from {pdf_path}, page {page_number}: {e}")
        return None


async def get_page_count_for_pdf(pdf_path):
    """Get page count for a pdf"""

    try:
        pdf_info = pdfinfo_from_path(pdf_path)
        return int(pdf_info['Pages'])
    except Exception as e:
        print(
            f"Error getting page count for {os.path.basename(pdf_path)}: {e}")


async def monitor_workers(total_count, data):
    """Shows worker status"""

    with tqdm(
        unit='pages', miniters=1,
        desc='Processed pages',
        total=total_count,
        leave=False
    ) as pbar:
        old = 0
        while len(data) < total_count:
            pbar.update(len(data) - old)
            old = len(data)
            pbar.refresh()

            await asyncio.sleep(0.2)


async def dataset_worker(q: asyncio.Queue, l: asyncio.Lock, title_indexes, data, labels):
    """Worker to create dataset for a pdf page"""

    while True:
        pdf_path, pdf_page = await q.get()

        image_data = await asyncio.to_thread(extract_image_features, pdf_path, pdf_page)
        if image_data is not None:
            async with l:
                data.append(image_data)
                if str(pdf_page) in title_indexes[pdf_path]:
                    # print(f'lable title: {pdf_page} {title_indexes[pdf_path]}')
                    labels.append("title")
                else:
                    # print(f'label normal {pdf_page} {title_indexes[pdf_path]}')
                    labels.append("normal")

        q.task_done()


async def execute_task(worker_task, parallelism, pdf_paths, data, start=1, end=0):
    """Execute tasks in parallel"""

    q = asyncio.Queue()
    l = asyncio.Lock()

    loop = asyncio.get_event_loop()

    if parallelism == 0:
        parallelism = os.cpu_count()

    workers = [loop.create_task(worker_task(q, l))
               for i in range(parallelism)]

    total_count = 0
    for pdf_path in pdf_paths:
        if not os.path.exists(pdf_path):
            print(f'File not found: {pdf_path}')
            continue

        if end == 0:
            num_pages = await get_page_count_for_pdf(pdf_path)
            if num_pages is None or num_pages == 0:
                continue
            total_count += num_pages
            current_end = num_pages
        else:
            current_end = end

        # print(f'{pdf_path}: {start}-{current_end}')

        for i in range(start, current_end + 1):
            await q.put((pdf_path, i))

    loop.create_task(monitor_workers(total_count, data))

    await q.join()

    for worker in workers:
        worker.cancel()

    await asyncio.gather(*workers, return_exceptions=True)


async def create_dataset(title_indexes, parallelism: int):
    """
    Creates a dataset of PDF pages as images.
    """
    data = []
    labels = []

    await execute_task(lambda q, l: dataset_worker(q, l, title_indexes, data, labels),
                       parallelism, title_indexes.keys(), data)

    return np.array(data), np.array(labels)


def create_cnn_model(input_shape):
    """
    Defines a simple Convolutional Neural Network (CNN) model.
    """
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(2, activation='softmax')  # Two classes: "title" and "normal"
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_and_evaluate_image_model(X, y):
    """
    Trains and evaluates the CNN model using image data.
    """
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    # Encode labels (e.g., "title" -> 0, "normal" -> 1)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)

    # Create and train the model
    input_shape = X_train.shape[1:]
    model = create_cnn_model(input_shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")

    # Detailed classification report
    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification Report:\n", classification_report(
        y_test, y_pred, target_names=label_encoder.classes_))

    return model, label_encoder


def predict_page_type_from_image(model, label_encoder,
                                 pdf_path, page_number,
                                 target_size=(128, 128)):
    """
    Predicts the page type (title or normal) for a new PDF page (as image).
    """
    image_data = extract_image_features(
        pdf_path, page_number, target_size=target_size)
    if image_data is None:
        return "Error: Could not extract image features."

    # The model expects a batch of images, so add a dimension
    image_data = np.expand_dims(image_data, axis=0)

    prediction = model.predict(image_data, verbose=0)
    predicted_class_index = np.argmax(prediction)
    predicted_label = label_encoder.inverse_transform([predicted_class_index])

    return predicted_label


async def predict_title_page_worker(q: asyncio.Queue, l: asyncio.Lock, pdf_title_pages,
                                    counts, model, label_encoder, image_target_size):
    """Worker to predict page category"""

    while True:
        pdf_path, pdf_page = await q.get()

        typ = await asyncio.to_thread(
            predict_page_type_from_image, model, label_encoder,
            pdf_path, pdf_page, image_target_size)

        counts.append(f'{pdf_path}-{pdf_page}')

        if typ == 'title':
            async with l:
                pdf_title_pages[pdf_path].append(pdf_page)

        q.task_done()


async def predict_title_pages(model, label_encoder, image_target_size,
                              pdf_paths, parallelism: int, start, end):
    """Predict pages"""

    pdf_title_pages = {}
    counts = []

    for pdf_path in pdf_paths:
        pdf_title_pages[pdf_path] = []

    await execute_task(lambda q, l: predict_title_page_worker(
        q, l, pdf_title_pages, counts, model, label_encoder, image_target_size),
        parallelism, pdf_paths, counts, start, end)

    return pdf_title_pages


async def create_model(args):
    """Create model"""

    title_indexes = {}

    save_path = args.save_path
    parallelism = args.parallelism

    for pdf_titles in args.pdf:
        title_indexes[pdf_titles[0]] = pdf_titles[1:]

    for pdf in args.files:
        title_indexes[pdf] = ['1']

    if parallelism == 0:
        parallelism = os.cpu_count()

    X_data, y_labels = await create_dataset(title_indexes, parallelism)
    print('Dataset created successfully.')
    model, label_encoder = train_and_evaluate_image_model(X_data, y_labels)
    save_data = {
        'model': model,
        'label': label_encoder,
    }

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)


async def predict_using_model(args):
    """Predict using model"""

    image_target_size = (128, 128)  # Consistent image size for the CNN
    load_path = args.model_path

    with open(load_path, 'rb') as f:
        load_data = pickle.load(f)

    model = load_data['model']
    label_encoder = load_data['label']

    start = args.beginning_page
    end = args.ending_page
    pdfs = args.pdf

    result = await predict_title_pages(model, label_encoder, image_target_size,
                                       pdfs, args.parallelism, start, end)

    return result


def overlay_help(image, replacement_mode, page, count):
    """Overlays help over image"""

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)  # Red color (BGR)
    font_thickness = 2
    all_overlay_texts = [[
        f'Title page/count: {page}/{count}',
        '->: next title page',
        '<-: previous title page',
        'x: delete current title page',
        'r: replace current title page',
        'n: next pdf file',
        's: save and quit',
        'q: quit without saving',
    ], [
        '->: next page',
        '<-: previous page',
        's: save current page as replacement',
        'q: quit page replacement mode',
    ]]

    overlay_texts = all_overlay_texts[1] if replacement_mode else all_overlay_texts[0]

    text_position = (50, 200)  # Bottom-left corner of the text
    for overlay_text in overlay_texts:
        cv2.putText(image, overlay_text, text_position,
                    font, font_scale, font_color,
                    font_thickness, cv2.LINE_AA)
        text_position = (text_position[0], text_position[1] + 50)


async def show_pages_as_images(pdf_titles):
    """Show pages detected as title as images"""
    for pdf_path, titles in pdf_titles.items():
        if not titles:
            print(f':: No titles for pages "{os.path.basename(pdf_path)}"')
            continue

        if len(titles) == 1:
            print(f':: Only one title page "{os.path.basename(pdf_path)}')

        page_count = await get_page_count_for_pdf(pdf_path)
        title_pages = []
        for t in titles:
            images = convert_from_path(
                pdf_path, first_page=t, last_page=t, dpi=100)
            title_pages.extend(images)

        window_title = f'Title pages from {re.sub("[^a-zA-Z0-9]", "", os.path.basename(pdf_path))}'

        replacement_mode = False
        replacement_page = -1

        image_index = 0
        while True:
            image = np.array(title_pages[image_index])
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            overlay_help(image, replacement_mode,
                         titles[image_index], len(titles))

            cv2.imshow(window_title, image)
            if replacement_mode:
                cv2.setWindowTitle(
                    window_title, f'Replacement page: {replacement_page}')
            else:
                start = titles[image_index]
                end = titles[image_index + 1] - \
                    1 if image_index + 1 < len(titles) else page_count
                cv2.setWindowTitle(
                    window_title, f'Title page for {start}-{end}')

            # Wait for a key press
            key = cv2.waitKey(0) & 0xFF

            if replacement_mode:
                if key == ord('d') or key == 83:
                    replacement_page += 1
                elif key == ord('a') or key == 81:
                    replacement_page -= 1
                elif key == ord('s'):
                    # Save changes and quit replacement mode
                    replacement_mode = False
                    titles[image_index] = replacement_page
                elif key == ord('q'):
                    # Quit replacement mode without save
                    replacement_mode = False
                    replacement_page = titles[image_index]

                new_images = convert_from_path(
                    pdf_path, first_page=replacement_page,
                    last_page=replacement_page, dpi=100)
                title_pages[image_index] = new_images[0]
            else:
                if key == ord('q'):
                    return pdf_titles, False
                if key == ord('s'):
                    return pdf_titles, True
                if key == ord('n'):
                    break
                elif key == ord('d') or key == 83:
                    image_index = (image_index + 1) % len(title_pages)
                elif key == ord('a') or key == 81:
                    image_index = (image_index - 1) % len(title_pages)
                elif key == ord('x'):
                    titles.pop(image_index)
                    title_pages.pop(image_index)
                    if not title_pages:
                        # No title page in current pdf
                        break
                    image_index = (image_index - 1) % len(title_pages)
                elif key == ord('r'):
                    replacement_mode = True
                    replacement_page = titles[image_index]

        cv2.destroyAllWindows()  # Close all OpenCV windows

    while True:
        action = input("No more files. Should save changes (yes/no)? ")
        if action.lower() == 'yes' or action.lower() == 'y':
            return pdf_titles, True
        elif action.lower() == 'no' or action.lower() == 'n':
            return pdf_titles, False

    return pdf_titles, False


async def save_split(dname, fname, title_pages, pdf, page_count, t, n, forceful, noop):
    """Save split file"""

    out_name = f'{dname}/{fname}_split_{n + 1:04d}.pdf'

    if not forceful and os.path.exists(out_name):
        print(f':: File "{os.path.basename(out_name)}" already exists')
        return

    start = title_pages[t] - 1 if t >= 0 else 0
    end = title_pages[t + 1] - 1 if t + \
        1 < len(title_pages) else page_count
    pdf_writer = PdfWriter()

    for i in range(start, end):
        pdf_writer.add_page(pdf.pages[i])

    if noop:
        print(f'::: Write pdf("{out_name}")')
    else:
        with open(out_name, 'wb') as f:
            pdf_writer.write(f)


async def split_pdf(pdf_path, title_pages, forceful, split_destination, noop):
    """Splits a PDF into individual page files."""

    dname = os.path.dirname(
        pdf_path) if split_destination is None else split_destination
    fname = os.path.splitext(os.path.basename(pdf_path))[0]
    pdf = PdfReader(pdf_path)
    page_count = await get_page_count_for_pdf(pdf_path)

    os.makedirs(dname, exist_ok=True)

    if title_pages[0] > 1:
        await save_split(dname, fname, title_pages, pdf, page_count, -1, -1, forceful, noop)

    for n, t in enumerate(range(len(title_pages))):
        await save_split(dname, fname, title_pages, pdf, page_count, t, n, forceful, noop)


async def split_pdfs(pdf_splits, forceful=False, move_original_to=None,
                     split_destination=None, move_singles=None, noop: bool = False):
    """Splits pdfs as per title pages"""

    msg = '' if split_destination is None else ''
    print(f':: Splitting pdfs{msg}')
    for pdf_path, title_pages in pdf_splits.items():
        original_name = os.path.basename(pdf_path)

        if not os.path.exists(pdf_path):
            print(f':: Skipping "{original_name}": file missing')
            continue

        if not title_pages:
            print(f':: Skipping "{original_name}": no title pages')
            continue

        if len(title_pages) == 1 and title_pages[0] == 1:
            if move_singles is None:
                print(f':: Skipping "{original_name}": split not required')
            else:
                move_path = f'{move_singles}/{os.path.basename(pdf_path)}'
                if noop:
                    print(f'::: Mkdir("{move_singles}")')
                    print(f'::: Rename("{pdf_path}", "{move_path}")')
                else:
                    os.makedirs(move_singles, exist_ok=True)
                    os.rename(pdf_path, move_path)
            continue

        print(f':: Splitting "{original_name}" ... ',
              flush=True, end='' if not noop else '\n')
        await split_pdf(pdf_path, title_pages, forceful, split_destination, noop)

        if move_original_to is not None:
            move_path = f'{move_original_to}/{original_name}'
            if noop:
                print(f'::: Mkdir("{move_original_to}")')
                print(f'::: Rename("{pdf_path}", "{move_path}")')
            else:
                os.makedirs(move_original_to, exist_ok=True)
                os.rename(pdf_path, move_path)

        if not noop:
            print('done')


async def main():
    """The main function"""

    args = parse_args()

    if args.command == 'create':
        await create_model(args)
    elif args.command == 'predict':
        result = await predict_using_model(args)
        with open(args.save_path, 'w', encoding="utf-8") as f:
            json.dump(result, f, indent=4)

        print(':: Detected titles:')
        for pdf_path, titles in result.items():
            print(f':: {pdf_path} -> {titles}')
    elif args.command == 'show':
        try:
            if args.show_command == 'run':
                result = await predict_using_model(args)
            elif args.show_command == 'from':
                with open(args.load_from_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            result, should_save = await show_pages_as_images(result)
            if should_save:
                with open(args.save_path, 'w', encoding="utf-8") as f:
                    json.dump(result, f, indent=4)
                print(f':: Saved split data into {args.save_path}')
            else:
                print(':: Skipped saving splits on user request')
        except (FileNotFoundError, PermissionError, IOError) as e:
            print(f'Error opening file ({args.load_from_file}): {e}')
    elif args.command == 'detect':
        try:
            if args.detect_command == 'run':
                result = await predict_using_model(args)
                await show_pages_as_images(result)
            elif args.detect_command == 'from':
                with open(args.load_from_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            await detect_title_page(args, result)

        except (FileNotFoundError, PermissionError, IOError) as e:
            print(f'Error opening file ({args.load_from_file}): {e}')
    elif args.command == 'split':
        try:
            if args.show_command == 'run':
                result = await predict_using_model(args)
            elif args.show_command == 'from':
                with open(args.load_from_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
            await split_pdfs(result, args.force, args.move_original_to,
                             args.split_destination, args.move_singles, args.noop)
        except (FileNotFoundError, PermissionError, IOError) as e:
            print(f'Error opening file ({args.load_from_file}): {e}')


MODEL_PATH = 'model.pkl'
TITLES_PATH = 'titles.json'
SPLITS_PATH = 'splits.json'


def add_create_command(commands):
    """Add create command arguments"""

    create_cmd = commands.add_parser('create', help='Create model')
    create_cmd.add_argument('-s', '--save-path', type=str, default=MODEL_PATH,
                            help=f'Save path (default: {MODEL_PATH})')
    create_cmd.add_argument('-p', '--parallelism', type=int, default=0,
                            help='Number of parallel pages to process (default: number of cores)')
    create_cmd.add_argument('files', type=str, nargs='+',
                            help='Pdf files to be used')
    create_cmd.add_argument('--pdf', nargs='+', action='append', metavar=('pdf', 'title-pages'),
                            help='Specify a file and comma separated title pages pair.'
                            ' Can be used multiple times.')


def add_predict_arguments(cmd, save_path=None):
    """Add predict command arguments"""

    cmd.add_argument('-m', '--model-path', type=str, default=MODEL_PATH,
                     help=f'Model file path (default: {MODEL_PATH})')
    if save_path is not None:
        cmd.add_argument('-s', '--save-path', type=str, default=save_path,
                         help=f'Save path (default: {save_path})')
    cmd.add_argument('-b', '--beginning-page', type=int, default=1,
                     help='Starting page of pdf file (default=1)')
    cmd.add_argument('-e', '--ending-page', type=int, default=0,
                     help='Ending page of pdf file (default=last page)')
    cmd.add_argument('-p', '--parallelism', type=int, default=0,
                     help='Number of parallel pages to process (default: number of cores)')
    cmd.add_argument('pdf', type=str, nargs='+',
                     help='Pdf files to be used')


def add_predict_command(commands):
    """Add predict command"""

    predict_cmd = commands.add_parser(
        'predict', help='Predict using a given model')
    add_predict_arguments(predict_cmd, TITLES_PATH)


def add_show_command(commands):
    """Add show command"""

    show_cmd = commands.add_parser(
        'show', help='Show pages predicted as title')
    show_commands = show_cmd.add_subparsers(
        dest='show_command', help='Available sub commands')
    run_cmd = show_commands.add_parser(
        'run', help='Run predict and show pages')
    add_predict_arguments(run_cmd)
    from_cmd = show_commands.add_parser(
        'from', help='Load saved data from file and show pages')
    from_cmd.add_argument('-l', '--load-from-file', type=str, default=TITLES_PATH,
                          help=f'Load title pages and pdf from file (default: {TITLES_PATH})')
    from_cmd.add_argument('-s', '--save-path', type=str, default=SPLITS_PATH,
                          help=f'Save path (default: {SPLITS_PATH})')


def add_split_arguments(cmd):
    """Add split arguments"""

    cmd.add_argument('--force', action='store_true',
                     help='Force overwriting of split files (default skips file if it exists)')
    cmd.add_argument('--move-original-to', type=str, default=None,
                     help='Post split move the file to specified diretory (default do not move)')
    cmd.add_argument('--split-destination', type=str, default=None,
                     help='Destination directory for split files (default same as source file)')
    cmd.add_argument('--noop', action='store_true',
                     help='Make no actual changes (default make changes)')
    cmd.add_argument('--move-singles', type=str, default=None,
                     help='Move files that contain only single title page and that too as first page'
                     ' into the specified directory (default is to not move)')


def add_split_command(commands):
    """Add split command"""

    split_cmd = commands.add_parser(
        'split', help='Split pages as per detected title pages')
    show_commands = split_cmd.add_subparsers(
        dest='show_command', help='Available sub commands')
    run_cmd = show_commands.add_parser(
        'run', help='Run predict and split pages')
    add_predict_arguments(run_cmd)
    add_split_arguments(run_cmd)
    from_cmd = show_commands.add_parser(
        'from', help='Load saved data from file and show pages')
    from_cmd.add_argument('-l', '--load-from-file', type=str, default=SPLITS_PATH,
                          help=f'Load title pages and pdf from file (default: {SPLITS_PATH})')
    add_split_arguments(from_cmd)


def parse_args():
    """Parses command line arguments"""

    parser = argparse.ArgumentParser(description='ai pdf splitter')

    commands = parser.add_subparsers(
        dest='command', help='Available commands', required=True)
    add_create_command(commands)
    add_predict_command(commands)
    add_show_command(commands)
    add_split_command(commands)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
