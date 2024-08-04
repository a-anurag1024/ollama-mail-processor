from mail_classifier.process_mail import MailProcessor, MailProcessor_Config

# initialize the processor
processor = MailProcessor(MailProcessor_Config())

# launch the processor
processor.run()