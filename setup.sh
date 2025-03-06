#Run these commands from terminal
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install pyenv
touch ~/.zshrc_custom
nano ~/.zshrc_custom
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv init -)"
sudo chmod +w ~/.zshrc
sudo sh -c 'echo "source ~/.zshrc_custom" >> ~/.zshrc'
sudo chmod -w ~/.zshrc
source ~/.zshrc
pyenv install 3.11.0
pyenv local 3.11.0
python3.11 -m venv venv
source venv/bin/activate
sh install.sh
echo "Environment setup complete. Run 'source venv/bin/activate' to activate it."