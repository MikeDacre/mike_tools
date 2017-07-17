#!/usr/bin/R


linearRegression <-
function(pd, gt, tmppfx, indiv) {

cat("Regression... "); flush.console()
options("scipen"=100)
n.indi = as.numeric(indiv)
print(n.indi)
postTotal = read.table(pd, sep="\t", header=T)
print(pd)
genofile = matrix(scan(file = gt, sep = "\t", what = double(), nlines = n.indi), byrow = TRUE, nrow = n.indi)
print(gt)
genoTotal = 1 - (0.5 * genofile)

# restrict to MAF cutoff
MAF= 0.01
post.start = 1
post.end=15
geno.start=16
geno.end=15 + n.indi

a = t(genoTotal)
b= cbind(postTotal,a)
mafsub = b[apply(b[,geno.start:geno.end], MARGIN = 1, function(x) mean(x) >= MAF & mean(x) <= 0.99), ]
post = mafsub[,1:15]
genotypes = as.matrix(mafsub[,geno.start:geno.end])
postTemp = mafsub


                                          
n.snps = nrow(post)
#genotypes=as.matrix(t(genos))
weights = rep(1/n.indi,n.indi)
real.props = genotypes %*% weights
depths=  post$Depth
postProp = post$POSTfreq
estimated.props = postProp
G = genotypes[,-n.indi] - genotypes[,n.indi]
Y = postProp - genotypes[,n.indi]
eps = 0.0001
# force postfreq to be between 0.0001 and 0.9999
props.for.weights = pmin(1-eps,pmax(estimated.props,eps))
# weight = depth / (adjusted.post * (1-adjusted.post))
regression.weights = depths / (props.for.weights * (1-props.for.weights) )
good = which( (postProp>0.1) & (postProp < 0.9))
m = lm(Y[good] ~ G[good,]-1,weights=regression.weights[good])  ## run without intercept
coefs = m$coef
s = summary(m)
cov.mat = s$cov.unscaled * s$sigma^2
big.cov.mat = matrix(NA,n.indi,n.indi)
big.cov.mat[-n.indi,-n.indi] = cov.mat
big.cov.mat[n.indi,n.indi] = sum(cov.mat)
big.cov.mat[n.indi,-n.indi] = big.cov.mat[-n.indi,n.indi] = -rowSums(cov.mat)
vars = sapply(1:n.snps, function(i) genotypes[i,] %*% big.cov.mat %*% genotypes[i,])
vars[vars < 0] <- 0 
all.coefs = c(coefs, 1-sum(coefs))
preProps = genotypes %*% all.coefs 
preProps[preProps > 1] <- 1 
preVars = vars 
postProps = estimated.props
postProps[postProps > 1] <- 1 
postVars = 1/regression.weights
postVars[postVars > 1] <- 1 
Zs = (postProps - preProps)/sqrt(preVars +postVars)

beta = (postProps - preProps)
vari = sqrt(preVars +postVars)
beta.table = cbind(beta, vari)
colnames(beta.table) <- c("beta", "variance")

write.table(beta.table, file = paste(tmppfx, ".betas.fmqtl", sep=""), sep="\t", quote=F, row.names=F, col.names=T)

}

linearRegression("ASW.20.POSTth.txt", "ASW.20.genotypes.txt", "ASW.20", "66")





