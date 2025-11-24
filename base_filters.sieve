require ["include", "environment", "variables", "relational", "comparator-i;ascii-numeric", "spamtest", "reject"];
require ["fileinto", "imap4flags", "envelope", "extlists", "vnd.proton.expire", "copy"];

# ============================================================
# IDENTIFY PROTECTED MESSAGES (MUST BE FIRST!)
# ============================================================

# Check if sender is in contacts or whitelist
if anyof(
  header :list "from" ":addrbook:personal",
  header :list "from" ":addrbook:myself",
  header :list "from" ":incomingdefaults:inbox"
) {
  addflag "Protected";
}

# Check for sensitive information that shouldn't be forwarded
if anyof(
    # Financial information
    header :contains "Subject" ["bank statement", "invoice", "payment", "SSN"],

    # Personal healthcare
    header :contains "Subject" ["medical", "prescription", "diagnosis"],

    # From known legitimate sources that might be false positives
    header :list "from" ":addrbook:personal",
    address :domain "From" ["yourcompany.com", "yourbank.com"]
) {
    addflag "sensitive";
}

# ============================================================
# SPECIFIC CATEGORIZATION
# ============================================================

# Receipts
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From" ["customerservice@tello.com"],
  header :comparator "i;unicode-casemap" :contains "Subject" "receiptarchiveme",
  address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] ["receipt"]
) {
  fileinto "Receipts";
  fileinto "archive";
  addflag ["\\Seen", "Protected"];
}

# System messages
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From" ["moodle"],
  header :comparator "i;unicode-casemap" :contains "Subject" ["Moodle"],
  address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] ["class@box.weefreemedic.org"]
) {
  fileinto "System";
  fileinto "archive";
  addflag ["\\Seen"];
}

# FM Lists
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From"
    ["stfmoffice@stfm.org", "DoNotReply@ConnectedCommunity.org"],
  header :comparator "i;unicode-casemap" :contains "Subject" ["FM Lists"],
  address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] ["fm@g.com"]
) {
  fileinto "Lists/FM";
  fileinto "archive";
}

# ============================================================
# HEADER CATEGORIZATION
# ============================================================

# Security tagging - unencrypted mail
if header :is "x-pm-transfer-encryption" "none" {
  fileinto "unencrypted";
  # Don't stop - allow further categorization
}

# Social media - auto-archive with expiry
if anyof(exists "x-facebook", exists "x-linkedin-id") {
  fileinto "archive";
  expire "day" "60";
  addflag "\\Seen";
}

# Mailing lists
if exists "list-unsubscribe" {
  fileinto "Mail List";
  fileinto "archive";
}

# ============================================================
# Message Protection
# ============================================================

# VIP handling with redundancy
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From" ["christina"]
) {
  addflag ["Protected", "\\Flagged", "Retain", "Important"];
  unexpire;
  keep;
}

if hasflag "Retain" {
  fileinto "inbox";
  unexpire;
  keep;
  stop;
}

if hasflag "Protected" {
  unexpire;
  keep;
  return;
}

# ============================================================
# Message DELETE/ARCHIVE
# ============================================================

# Archive
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From" ["email@heartemail.org"],
  header :comparator "i;unicode-casemap" :contains "Subject" ["archiveme"],
  address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] ["archiveme"]
) {
  fileinto "archive";
  addflag "\\Seen";
}

# Trash - direct to trash folder
if anyof(
  address :all :comparator "i;unicode-casemap" :contains "From" ["no-reply@tinyurl.email"],
  header :comparator "i;unicode-casemap" :contains "Subject" "deleteme",
  address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] "badder@faith.com"
) {
  fileinto "trash";
  expire "day" "7";  # Auto-delete trash after a week
}


# ============================================================
# DESTRUCTIVE ACTIONS (SPAM/REJECT/DISCARD)
# Only execute if NOT protected
# ============================================================

if hasflag "Protected" {
  unexpire;
  keep;
  return;

} else {
  # Reject
  if anyof(
    address :all :comparator "i;unicode-casemap" :contains "From" ["unsubscribed"],
    header :comparator "i;unicode-casemap" :contains "Subject" "unsub",
    address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] "unsub"
  ) {
    addflag ["reject", "report", "junk"];
  }

  # Silent discard
  if anyof(
    address :all :comparator "i;unicode-casemap" :contains "From" ["epicearth"],
    header :comparator "i;unicode-casemap" :contains "Subject" "drop",
    address :all :comparator "i;unicode-casemap" :contains ["To", "Cc", "Bcc"] "breject",
    header :list "from" ":incomingdefaults:spam"
  ) {
    discard;
  }

  # Spam handling - both automatic and manual
  if anyof(
      address :all :comparator "i;unicode-casemap" :contains "From" ["abmgood", "hologic", "syncell", "exactsciences"],
      allof(
       environment :matches "vnd.proton.spam-threshold" "*",
       spamtest :value "ge" :comparator "i;ascii-numeric" "8"
     )
  ) {
    addflag ["\\Seen", "junk", "report"];
    expire "hour" "1";
  } elsif allof(
       environment :matches "vnd.proton.spam-threshold" "*",
       spamtest :value "ge" :comparator "i;ascii-numeric" "3"
     ) {
       addflag ["\\Seen", "junk"];
  }

  if hasflag "junk" {
    fileinto "spam";
  }

  # Report Crap Messages
  if hasflag "report" {
    if anyof (
      # header :exists "X-Spam-Report-Loop",
      header :contains "Received" ["spam.spamcop.net", "knujon.net"],
      envelope :all :is "from" ["spamcop_addr", "knujon_addr"]
    ) {
      # This is already a forwarded/processed message
      discard;
      stop;
    } elsif hasflag ["sensitive", "Protected"] {
      unexpire;
      keep;
    } elsif hasflag "reported" {
      stop;
    } else {
      # Forward to Spam Lists
      # redirect :copy "spamcop_addr";
      addflag "reported";
    }
  }

  # Finally Reject
  if hasflag "reject" {
    reject "Unsubscribe";
    stop;
  }
}

