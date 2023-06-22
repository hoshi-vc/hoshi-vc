# https://hub.docker.com/layers/nvidia/cuda/12.1.1-cudnn8-devel-ubuntu22.04/images/sha256-41e87805185deb6904fa4c6a0638add0f8ed098d070958208de945e5433732b4
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV LANG C.UTF-8
ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

# Based on https://github.com/gradient-ai/base-container/blob/main/all-pt200-tf212-jax048-py311/Dockerfile
# 1st line: CLI Apps and Project dependencies
# 2nd line: Python dependencies -- https://devguide.python.org/getting-started/setup-building/#linux
RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  curl espeak-ng fd-find ffmpeg git htop ncdu neovim nnn ripgrep rsync sudo unzip wget zip \
  build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma lzma-dev tk-dev uuid-dev zlib1g-dev

RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.11.3 && \
  echo -e '\n. ~/.asdf/asdf.sh' >>~/.bashrc && \
  echo -e '\n. ~/.asdf/completions/asdf.bash' >>~/.bashrc

RUN git clone --depth 1 https://github.com/hoshi-vc/hoshi-vc && \
  cp hoshi-vc/.tool-versions ~/.tool-versions && \
  rm -rf hoshi-vc

RUN bash -c "source ~/.asdf/asdf.sh && asdf plugin add python && asdf install python"

RUN bash -c "source ~/.asdf/asdf.sh && python3 -m pip --no-cache-dir install --upgrade pip jupyterlab jupyterlab-snippets jupyter_contrib_nbextensions jupyterlab-git && asdf reshim" && \
  bash -c "source ~/.asdf/asdf.sh && jupyter contrib nbextension install --user"

RUN bash -c "source ~/.asdf/asdf.sh && asdf plugin add pdm && asdf install pdm"
RUN bash -c "source ~/.asdf/asdf.sh && asdf plugin add nodejs && asdf install nodejs"

# TODO: respect the version in requirements.txt
# Nightly PyTorch causes torchaudio.load() to segfault, so ignore the version as a easy workaround for now.
RUN git clone --depth 1 https://github.com/hoshi-vc/hoshi-vc && \
  cat hoshi-vc/requirements.txt | sed 's/\(.*\)==.*/\1/' | sed '/^-/d' | sed '/^#/d' > ~/requirements.txt && \
  rm -rf hoshi-vc && \
  bash -c "source ~/.asdf/asdf.sh && pdm config check_update false" && \
  bash -c "source ~/.asdf/asdf.sh && pip install --no-cache-dir -r ~/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118 && asdf reshim" && \
  rm ~/requirements.txt

RUN curl -L "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" -o ~/code.tar.gz && \
  tar -xzf ~/code.tar.gz -C ~ && \
  mv ~/code /usr/local/bin/code && \
  rm ~/code.tar.gz

RUN curl -L "https://github.com/coder/code-server/releases/download/v4.13.0/code-server_4.13.0_amd64.deb" -o ~/code-server.deb && \
  apt-get install -y --no-install-recommends ~/code-server.deb && \
  rm ~/code-server.deb && \
  mkdir -p ~/.vscode-server/extensions && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension antfu.unocss && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension astro-build.astro-vscode && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension bungcip.better-toml  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension EditorConfig.EditorConfig  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension esbenp.prettier-vscode  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension foxundermoon.shell-format  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension Gruntfuggly.todo-tree jock.svg  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension meganrogge.template-string-converter  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension mhutchie.git-graph  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension moalamri.inline-fold  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension ms-python.isort  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension ms-python.python  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension ms-toolsai.jupyter  && \
  code-server --extensions-dir ~/.vscode-server/extensions --install-extension xyz.local-history && \
  apt-get remove -y --purge code-server && \
  apt-get autoremove -y && \
  rm -rf ~/.local/share/code-server ~/.config/code-server

ADD vocoder.tar /root/data/
ADD datasets.tar /root/data/
ADD feats.tar /root/data/
ADD attempt07-stage1.tar /root/data/

EXPOSE 8888 6006

CMD bash -c "source ~/.asdf/asdf.sh && jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True"
