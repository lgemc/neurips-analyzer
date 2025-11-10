.PHONY: help upload-db create-web-db all

help:
	@echo "NeurIPS Database Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  create-web-db  - Create web-optimized database from docs/neurips.db"
	@echo "  upload-db      - Upload web-optimized database to S3"
	@echo "  all            - Create and upload web-optimized database"
	@echo "  help           - Show this help message"

create-web-db:
	@echo "Creating web-optimized database..."
	python pipeline/step6_web_db.py --input neurips.db --output neurips_web.db
	@echo "Compressing database with gzip..."
	gzip -9 -k -f neurips_web.db
	@echo "✓ Created neurips_web.db.gz (compressed)"

upload-db:
	@echo "Uploading compressed web-optimized database to S3..."
	aws s3 cp neurips_web.db.gz s3://lgemc-static/neurips.db.gz --acl public-read --content-encoding gzip --content-type application/x-sqlite3
	@echo "✓ Database uploaded to: https://lgemc-static.s3.amazonaws.com/neurips.db.gz"

all: create-web-db upload-db
	@echo "✓ Complete! Web-optimized database created and uploaded."
