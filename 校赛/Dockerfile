FROM public.ecr.aws/docker/library/python@sha256:1475c4f5b6bcb7d35e57661d6f5d7429ec4041275633c15f2b49350ac898276d
WORKDIR /app
RUN pip install --no-cache-dir numpy==2.2.6
COPY work/sdk/contest_harness /app/contest_harness
COPY work/sdk/template/harness.py /app/harness.py
COPY app/*.py /app/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
USER 65534:65534
ENTRYPOINT ["python3","/app/harness.py"]
CMD ["describe"]
