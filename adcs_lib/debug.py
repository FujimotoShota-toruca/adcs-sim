
def debug_point():
    user_input = input("続行しますか？ (y/n): ").strip().lower()
    if user_input == 'y':
        print("続行します。")
        # ここに処理を記述
    elif user_input == 'n':
        print("停止します。")
        exit()