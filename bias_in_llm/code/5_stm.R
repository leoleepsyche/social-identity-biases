library(stm)
library(ggplot2)

alldata <- read.csv('./data/all_data_berttopic.csv', stringsAsFactors = FALSE)

# 1. Prepare documents and metadata
processed <- textProcessor(
  documents = alldata$sentence_segmented,  # Your tokenized Chinese text
  metadata = alldata,
  removestopwords = TRUE,
  customstopwords = c("我们是", "他们是"),
  removepunctuation = TRUE,
  removenumbers = TRUE,
  stem = FALSE,  
  wordLengths = c(1,15),
  sparselevel = 1,
  verbose = TRUE,
  onlycharacter = TRUE,
  striphtml = TRUE
)

# 2. Prepare data for modeling
out <- prepDocuments(
  processed$documents,
  processed$vocab,
  processed$meta,
  lower.thresh = 20  # Only keep words that occur at least 20 times
)

# #### 3. Model Selection ####
# kResult <- searchK(out$documents, out$vocab,
#                    K = c(20, 40, 60, 80),
#                    data = out$meta
# )
# 
# pdf("kplot.pdf")
# plot(kResult)
# dev.off()

# 4. Train the STM:
set.seed(17)
stm_model <- stm(
  documents = out$documents,
  vocab = out$vocab,
  K = 60,
  data = out$meta
)

# 5. Get the dominant topic for each document:
dominant_topics <- max.col(as.data.frame(stm_model$theta))

# 6. Now we can align by the `id` in the metadata:
results <- data.frame(id = out$meta$id, dominant_topic = dominant_topics)

# 7. Then merge with the original data:
final <- merge(alldata, results, by = "id", all.x = TRUE)

# 8. Save to CSV:
write.csv(final, "./data/all_data_berttopic_stm.csv", row.names = FALSE)


