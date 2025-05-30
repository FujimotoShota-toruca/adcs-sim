from exit_output import discord

if __name__ == "__main__":
    discord_utility = discord.DiscordSendMessage()

    # 完了通知をDiscordへ送信
    discord_utility.send_message("通知のテストです。")