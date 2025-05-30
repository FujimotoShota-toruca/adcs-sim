
def update_progress_bar(current, total):
    """
    進捗バーを表示する関数

    Parameters:
    - current (int): 現在の進捗
    - total (int): 総進捗
    """
    progress = current / total
    bar_length = 50
    block = int(round(bar_length * progress))
    progress_bar = "#" * block + "-" * (bar_length - block)
    print(f"\r[{progress_bar}] {current}/{total} ({progress * 100:.2f}%)", end='', flush=True)
