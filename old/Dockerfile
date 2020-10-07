FROM allennlp/allennlp:v1.0.0rc4

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev wget curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /src

COPY ./adat /src/adat
COPY ./bin /src/bin
COPY ./configs /src/configs
COPY ./scripts /src/scripts
COPY ./datasets /src/datasets

# copy NLP models
COPY ./logs/nlp/lm/model.tar.gz /src/logs/nlp/lm/
COPY ./logs/nlp/lev/model.tar.gz /src/logs/nlp/lev/

COPY ./logs/nlp/dataset_ag/target_clf/model.tar.gz /src/logs/nlp/dataset_ag/target_clf/
COPY ./logs/nlp/dataset_ag/substitute_clf/model.tar.gz /src/logs/nlp/dataset_ag/substitute_clf/

COPY ./logs/nlp/dataset_sst/target_clf/model.tar.gz /src/logs/nlp/dataset_sst/target_clf/
COPY ./logs/nlp/dataset_sst/substitute_clf/model.tar.gz /src/logs/nlp/dataset_sst/substitute_clf/

COPY ./logs/nlp/dataset_trec/target_clf/model.tar.gz /src/logs/nlp/dataset_trec/target_clf/
COPY ./logs/nlp/dataset_trec/substitute_clf/model.tar.gz /src/logs/nlp/dataset_trec/substitute_clf/

COPY ./logs/nlp/dataset_mr/target_clf/model.tar.gz /src/logs/nlp/dataset_mr/target_clf/
COPY ./logs/nlp/dataset_mr/substitute_clf/model.tar.gz /src/logs/nlp/dataset_mr/substitute_clf/

# copy non-NLP models
COPY ./logs/non_nlp/dataset_age/target_clf/model.tar.gz /src/logs/non_nlp/dataset_age/target_clf/
COPY ./logs/non_nlp/dataset_age/substitute_clf/model.tar.gz /src/logs/non_nlp/dataset_age/substitute_clf/
COPY ./logs/non_nlp/dataset_age/lm/model.tar.gz /src/logs/non_nlp/dataset_age/lm/
COPY ./logs/non_nlp/dataset_age/lev/model.tar.gz /src/logs/non_nlp/dataset_age/lev/

COPY ./logs/non_nlp/dataset_gender/target_clf/model.tar.gz /src/logs/non_nlp/dataset_gender/target_clf/
COPY ./logs/non_nlp/dataset_gender/substitute_clf/model.tar.gz /src/logs/non_nlp/dataset_gender/substitute_clf/
COPY ./logs/non_nlp/dataset_gender/lm/model.tar.gz /src/logs/non_nlp/dataset_gender/lm/
COPY ./logs/non_nlp/dataset_gender/lev/model.tar.gz /src/logs/non_nlp/dataset_gender/lev/

COPY ./logs/non_nlp/dataset_ins/target_clf/model.tar.gz /src/logs/non_nlp/dataset_ins/target_clf/
COPY ./logs/non_nlp/dataset_ins/substitute_clf/model.tar.gz /src/logs/non_nlp/dataset_ins/substitute_clf/
COPY ./logs/non_nlp/dataset_ins/lm/model.tar.gz /src/logs/non_nlp/dataset_ins/lm/
COPY ./logs/non_nlp/dataset_ins/lev/model.tar.gz /src/logs/non_nlp/dataset_ins/lev/


COPY pyproject.toml poetry.lock readme.md /src/
RUN pip install --user --upgrade pip \
    && pip install poetry==0.12.17 --no-cache-dir \
    && poetry config settings.virtualenvs.create false \
    && poetry install